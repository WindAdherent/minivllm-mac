# MLX Attention on macOS Design

## Context

`src/myvllm/layers/attention.py` currently exposes a Torch-based inference interface, but its working-tree implementation depends on Triton and CUDA kernels. Triton/CUDA is not the target execution path for this Apple Silicon project. At the same time, callers in `layers/` and `models/qwen3.py` pass `torch.Tensor` values and expect `Attention` to remain a `torch.nn.Module`.

MLX 0.32.0 is installed in the project environment. Metal, DLPack interoperation, and `mx.fast.scaled_dot_product_attention` are available on the target Mac.

## Goals

- Replace Triton/CUDA attention execution with native MLX operations on macOS.
- Preserve every public function name, positional argument, return shape, dtype, device, and observable side effect.
- Continue accepting and returning `torch.Tensor` objects.
- Preserve in-place updates to the caller-owned K/V cache tensors.
- Preserve variable-length causal prefill, grouped-query attention, and paged decode behavior.
- Remove actual CUDA calls from `src/myvllm/layers/` and `src/myvllm/models/qwen3.py`.
- Keep CPU execution usable for deterministic unit tests.

## Non-goals

- Migrating the whole model from `torch.nn.Module` to `mlx.nn.Module`.
- Changing tensor-parallel or distributed behavior. Existing `torch.distributed` usage remains because it is platform-independent and the repository already uses the macOS-compatible Gloo backend in its example setup.
- Adding training or autograd support. This project path is inference-oriented, and the current Triton implementation does not provide a Torch autograd contract.
- Refactoring unrelated layers or model architecture.

## Public compatibility contract

The signatures of `store_kvcache`, `flash_attention_prefill`, `paged_attention_decode`, `Attention.__init__`, and `Attention.forward` remain unchanged. `Attention` continues to inherit from `torch.nn.Module`.

For valid existing inputs:

- `store_kvcache` mutates the same `k_cache` and `v_cache` objects supplied by the caller, maps non-negative slots to `(block_index, block_offset)`, and ignores `-1` entries.
- `flash_attention_prefill` returns `(total_tokens, num_heads, head_dim)` on the same Torch device and with the same dtype as `q`. Each packed sequence is independently causal, and GQA maps query heads to their corresponding KV heads.
- `paged_attention_decode` returns `(batch_size, num_heads, head_dim)` on the query device and dtype. It follows each row of `block_tables`, respects `context_lens`, supports GQA, and returns zeros for an empty context.
- `Attention.forward` retains the current prefill/decode dispatch, cache update behavior, scaling convention, and flattened final dimension.

## Architecture

### Torch/MLX interoperability helpers

Private helpers convert contiguous Torch tensors to MLX through DLPack and convert evaluated MLX arrays back through DLPack. Before exposing an MPS tensor to MLX, the helper synchronizes pending Torch MPS work; before returning an MLX result to Torch, it evaluates the MLX graph. This prevents asynchronous producer/consumer races while avoiding host copies on Metal.

CPU tensors use the same public helpers. Where DLPack storage is not shared, the cache path explicitly copies the MLX result back into the original Torch cache tensor so object identity and mutation semantics remain intact.

### KV cache storage

`store_kvcache` removes the Triton kernel. It converts tensors and slot mappings to MLX, filters `-1` slots, derives block indices and offsets with MLX integer operations, and performs indexed MLX updates for keys and values. The graph is evaluated before returning. On Metal, the DLPack-backed cache uses shared storage; on CPU, updated values are copied back into the original cache tensors.

The existing cache-shape and slot-count assertions remain. Empty input or an all-`-1` mapping is a no-op.

### Variable-length prefill

Packed sequence boundaries are read from `cu_seqlens` in their existing order. For each non-empty sequence, Q/K/V are transposed into MLX's `[batch, heads, tokens, head_dim]` layout and passed to `mx.fast.scaled_dot_product_attention` with the supplied `scale` and `mask="causal"`.

MLX natively supports grouped-query attention, so K/V heads are not repeated. Per-sequence results are restored to the existing token-major layout and concatenated. An empty packed input returns an empty tensor matching `q`.

### Paged decode

For each batch row, MLX derives logical token positions, maps them through `block_tables`, and gathers the required K/V rows from the paged cache. The single-token query and gathered K/V tensors are transformed to MLX attention layout and passed to `mx.fast.scaled_dot_product_attention` without a causal mask, since every gathered cache token is visible during decode.

An empty context produces zeros for that batch row. Batch results are concatenated and converted back to a Torch tensor matching the query's device and dtype.

### `Attention` integration

`Attention.forward` stays a Torch-facing adapter. It keeps the existing 3-D/4-D K/V normalization, context validation, cache dispatch, scaling, prefill/decode branching, and output reshape. The initial empty caches remain Torch tensors so existing model code and cache allocation code do not change.

## macOS cleanup outside the kernels

- Remove Triton imports, decorators, and kernels from `attention.py`.
- Update the `attention.py` example from `.cuda()` and `torch.cuda.synchronize()` to MPS equivalents, or simplify it if required to keep the example valid.
- Move both the model and input in the `qwen3.py` example to MPS instead of moving only the input to CUDA.
- Leave existing MPS examples in `activation.py` and `layernorm.py` unchanged.
- Leave `torch.distributed` code unchanged; it is not a CUDA dependency.

## Error handling and edge cases

- Retain existing assertions for mismatched K/V cache shapes and slot count.
- Keep the existing `ValueError` when prefill context lacks `cu_seqlens_q`.
- Skip `-1` cache slots without modifying their cache locations.
- Return zero decode output for `context_len == 0` rather than dividing by an empty softmax normalizer.
- Evaluate MLX graphs before Torch consumes their outputs or observes cache mutations.
- Avoid silently moving public outputs to a different device or dtype.

## Testing strategy

Implementation follows red-green-refactor:

1. Restore/add focused behavior tests before changing production code and observe them fail against the Triton/CUDA implementation.
2. Test module import without Triton.
3. Test cache slot mapping, `-1` skipping, cache identity, and in-place mutation.
4. Compare variable-length causal GQA prefill against a small explicit Torch reference.
5. Compare paged GQA decode against a small explicit Torch reference, including empty context.
6. Test `Attention.forward` shape and dispatch compatibility through the existing context interface where practical.
7. Add a Metal integration test, guarded by MPS availability, to exercise DLPack sharing and MLX execution on the target device.
8. Scan the requested source scope for executable Triton/CUDA usage.
9. Run the focused attention tests and then the complete uv-managed pytest suite outside the restricted sandbox so Metal is accessible.

## Risks and mitigations

- **Cross-framework synchronization:** Explicit Torch MPS synchronization and MLX evaluation establish producer/consumer ordering.
- **Cache mutation across DLPack:** Tests verify both Metal shared-storage behavior and CPU copy-back behavior while preserving the original Torch object identity.
- **Numerical differences:** Tests use tolerances appropriate for the input dtype and compare against the previous mathematical contract, not Triton implementation details.
- **Performance loss from host copies:** Metal data stays on-device through DLPack; only small sequence metadata is materialized for Python control flow.
- **Unrelated dirty worktree content:** Only scoped files and new tests/docs are edited; the pre-existing `attention.py` changes are treated as the user's source state and are not discarded through Git reset or checkout.
