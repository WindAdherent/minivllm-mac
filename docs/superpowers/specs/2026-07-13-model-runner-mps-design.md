# ModelRunner on macOS MPS Design

## Context

`src/myvllm/engine/model_runner.py` currently presents a Torch-based model-running interface but assumes CUDA throughout initialization, tensor placement, memory accounting, decode graph capture, and cleanup. The repository targets Apple Silicon, and its attention layer already preserves the Torch interface while using MPS and, where useful, MLX internally.

The target machine provides PyTorch 2.12 with MPS, MLX on Metal, and the Gloo distributed backend. NCCL is unavailable. The current runtime does not use distributed inference, so this change explicitly supports only `world_size == 1`.

## Goals

- Make `ModelRunner` execute on Apple Silicon through PyTorch MPS.
- Remove executable CUDA and NCCL dependencies from `model_runner.py`.
- Preserve the existing model, sampling, sequence, context, and KV-cache behavior visible to callers.
- Preserve the existing configuration keys and the Torch tensor interface.
- Keep Qwen3 and Llama model selection and checkpoint loading behavior unchanged.
- Keep the KV-cache layout and `config['max_cached_blocks']` contract unchanged.
- Verify the implementation with the project's uv-managed environment on the target Mac.

## Non-goals

- Supporting `world_size > 1` or tensor-parallel execution on MPS.
- Rewriting Torch model classes as MLX models.
- Changing model architecture, attention semantics, sampling behavior, scheduler behavior, or checkpoint format.
- Introducing MPS compilation or an unproven replacement for CUDA Graph capture.
- Refactoring unrelated files beyond focused tests and implementation documentation.

## Compatibility contract

For valid single-process configurations, construction and inference retain their current observable contract:

- `ModelRunner(config, rank, event)` accepts the same arguments. Valid macOS execution requires `world_size == 1` and `rank == 0`.
- The configured Qwen3 or Llama model is constructed, moved to the accelerator, and loaded through the existing checkpoint loader.
- `prepare_prefill`, `prepare_decode`, and `prepare_sample` return Torch tensors with their existing shapes and dtypes, now on MPS.
- `run_model` returns logits from the existing model and `compute_logits` methods.
- `run` samples only on rank zero, resets inference context, and returns the same token-id form.
- KV caches retain the layout `(2, num_layers, max_cached_blocks, block_size, num_kv_heads, head_dim)` and are installed into each attention module's `k_cache` and `v_cache` attributes.
- `exit` releases owned resources and destroys the process group when initialized.

Configurations with `world_size != 1` fail early with a clear error instead of reaching an unavailable NCCL or unsupported MPS tensor-parallel path.

## Architecture

### Device and process-group initialization

`ModelRunner` owns a `torch.device('mps')` value and uses it for all model and tensor placement. Construction validates MPS availability, `world_size == 1`, and `rank == 0` before model allocation. The process group uses Gloo with the existing local initialization endpoint because model-layer constructors query `torch.distributed` for rank and world size even in the single-process case.

The existing `enforce_eager` configuration key remains accepted. MPS execution is always eager because CUDA Graphs have no direct MPS equivalent in this design; therefore the key becomes a performance hint with no effect on numerical behavior.

### Model and input placement

The model is moved with `model.to(self.device)` before checkpoint loading, preserving the existing order. Input IDs, sequence metadata, cache slot mappings, context lengths, block tables, and temperatures are constructed directly on MPS or moved with `.to(self.device)`.

Pinned host allocation and CUDA-specific nonblocking copies are removed. They are not part of the public contract and do not provide the same transfer path for the unified-memory MPS backend.

### Warmup and memory accounting

Warmup continues to execute the largest configured synthetic prefill before KV-cache allocation. It uses `torch.mps.empty_cache()` around the run and `torch.mps.synchronize()` before memory measurements so queued Metal work is reflected in the accounting.

MPS has no CUDA-style peak-memory reset/statistics contract. The KV-cache budget is therefore computed from the device's recommended working-set limit:

```text
memory_limit = recommended_max_memory * gpu_memory_utilization
available_for_cache = memory_limit - driver_allocated_memory
```

`driver_allocated_memory` is used because it includes current framework/device allocations that compete for the Metal working set. The existing block-size calculation then converts the available byte budget into `max_cached_blocks`. A non-positive budget or fewer than one block raises the existing not-enough-memory assertion with rank context.

### KV-cache allocation

The cache tensor is created on MPS with the existing zero-initialized shape and default dtype. Each attention module receives views into the correct layer of the shared allocation. Since only `world_size == 1` is supported, cross-rank minimum reduction is removed; the locally computed block count is the global block count for the scheduler.

### Eager prefill and decode

Both prefill and decode call the model eagerly, followed by `compute_logits`. CUDA Graph capture, replay buffers, graph pools, and graph cleanup are removed. This changes only the optimization path: input/context preparation, attention cache updates, logits, sampling, and return values remain the same.

The internal graph-capture method is removed because it has no macOS implementation and is not part of the documented external engine contract. The `enforce_eager` configuration key remains accepted to avoid breaking existing configuration dictionaries.

### Cleanup

Cleanup uses `torch.mps.synchronize()` instead of CUDA synchronization. Shared-memory cleanup remains structurally present but is unreachable for the supported `world_size == 1` configuration. The initialized Gloo process group is destroyed as before.

## Error handling and edge cases

- Raise a clear runtime error when MPS is unavailable.
- Raise a clear value error when `world_size != 1` or `rank != 0`.
- Retain the unsupported-model exception and checkpoint-loading errors.
- Reject a KV-cache budget that cannot hold at least one block.
- Preserve empty optional block-table handling during prefill.
- Keep context reset behavior after successful inference.
- Do not silently fall back to CPU, because that would hide a deployment error and materially change performance behavior.

## Testing strategy

Implementation follows red-green-refactor:

1. Add source-level regression coverage proving `model_runner.py` has no executable CUDA device calls, CUDA memory calls, CUDA Graph use, or NCCL initialization.
2. Add focused tests for the early `world_size` and rank validation.
3. Exercise `prepare_prefill`, `prepare_decode`, and `prepare_sample` on MPS and verify shapes, dtypes, devices, and context fields.
4. Exercise eager `run_model` for both prefill and decode while `enforce_eager` is false, proving decode no longer expects captured graphs.
5. Isolate KV-cache accounting with controlled MPS memory values and verify block count, cache shape, zero initialization, and per-layer cache assignment.
6. Run an MPS smoke path through the existing model-facing interfaces where practical without requiring external checkpoint downloads.
7. Run the focused model-runner tests and the complete `uv run --no-sync pytest` suite with Metal access.
8. Scan the final diff and requested source file to ensure unrelated user changes are preserved.

## Risks and mitigations

- **Unified-memory over-allocation:** Base the cache pool on MPS's recommended working-set size and current driver allocation, then apply the existing utilization factor.
- **Asynchronous memory readings:** Synchronize MPS before reading device allocation counters.
- **Decode performance regression:** Accept eager decode as the stable MPS baseline; reusable buffers or compilation can be evaluated separately with benchmarks.
- **Configuration ambiguity:** Keep `enforce_eager` accepted and document that MPS always executes eagerly.
- **Distributed misuse:** Reject unsupported ranks and world sizes before allocating the model or initializing a process group.
- **Dirty working tree:** Treat the existing 458-line `model_runner.py` worktree content as user-owned source and apply only scoped patches without reset or checkout.
