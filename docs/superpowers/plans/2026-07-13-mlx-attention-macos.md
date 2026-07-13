# MLX Attention on macOS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Triton/CUDA attention implementation with native MLX execution while preserving the Torch-facing API and making both `attention.py` and `qwen3.py` main programs runnable through uv on Apple Silicon.

**Architecture:** Keep `Attention` as a `torch.nn.Module` and introduce private DLPack adapters that synchronize Torch MPS producers, execute MLX graphs on Metal, and return Torch tensors on the caller's original device and dtype. Implement prefill and decode with `mx.fast.scaled_dot_product_attention`, use MLX indexed cache updates with CPU copy-back, and limit non-attention edits to macOS-safe main programs.

**Tech Stack:** Python 3.11, PyTorch 2.12, MLX 0.32, DLPack, pytest 9.1, uv, Apple Metal/MPS.

---

## File map

- Create `tests/test_attention.py`: compatibility regression tests and the executable CUDA/Triton source scan.
- Modify `src/myvllm/layers/attention.py`: Torch/MLX adapters, MLX cache/prefill/decode implementations, unchanged public class contract, and a runnable MPS main smoke test.
- Modify `src/myvllm/models/qwen3.py`: explicit distributed import and a small Gloo + MPS prefill main smoke test.
- Modify this plan only to check off completed steps while executing.

### Task 1: Restore the compatibility regression suite

**Files:**
- Create: `tests/test_attention.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Write the failing import and behavior tests**

Create a test module that loads `attention.py` directly with a stubbed `myvllm.utils`, defines an explicit GQA reference, and includes these tests:

```python
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ATTENTION_PATH = ROOT / "src" / "myvllm" / "layers" / "attention.py"


def load_attention_module():
    utils = types.ModuleType("myvllm.utils")
    utils.get_context = lambda: None
    spec = importlib.util.spec_from_file_location("attention_under_test", ATTENTION_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"myvllm.utils": utils}):
        spec.loader.exec_module(module)
    return module


def reference_attention(query, key, value, scale, causal):
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    queries_per_kv_head = num_heads // num_kv_heads
    heads = []
    for head_index in range(num_heads):
        kv_head_index = head_index // queries_per_kv_head
        scores = query[:, head_index] @ key[:, kv_head_index].transpose(0, 1)
        scores = scores * scale
        if causal:
            mask = torch.ones_like(scores, dtype=torch.bool).triu(diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        heads.append(torch.softmax(scores, dim=-1) @ value[:, kv_head_index])
    return torch.stack(heads, dim=1)


def test_attention_module_imports_without_triton():
    module = load_attention_module()
    assert module.Attention is not None


def test_store_kvcache_maps_slots_and_skips_negative_entries():
    module = load_attention_module()
    key = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    value = key + 100
    k_cache = torch.full((3, 2, 2, 3), -1.0)
    v_cache = torch.full_like(k_cache, -1.0)
    k_identity, v_identity = id(k_cache), id(v_cache)
    module.store_kvcache(
        key, value, k_cache, v_cache, torch.tensor([0, 3, -1, 5]), block_size=2
    )
    expected_k = torch.full_like(k_cache, -1.0)
    expected_v = torch.full_like(v_cache, -1.0)
    expected_k[0, 0], expected_k[1, 1], expected_k[2, 1] = key[0], key[1], key[3]
    expected_v[0, 0], expected_v[1, 1], expected_v[2, 1] = value[0], value[1], value[3]
    assert id(k_cache) == k_identity
    assert id(v_cache) == v_identity
    torch.testing.assert_close(k_cache, expected_k)
    torch.testing.assert_close(v_cache, expected_v)


def test_flash_attention_prefill_handles_varlen_causal_gqa():
    module = load_attention_module()
    torch.manual_seed(7)
    q, k, v = torch.randn(5, 4, 3), torch.randn(5, 2, 3), torch.randn(5, 2, 3)
    output = module.flash_attention_prefill(
        q, k, v, torch.tensor([0, 2, 5]), 0.37, 4, 2, 3
    )
    expected = torch.cat(
        [
            reference_attention(q[:2], k[:2], v[:2], 0.37, causal=True),
            reference_attention(q[2:], k[2:], v[2:], 0.37, causal=True),
        ]
    )
    assert output.device == q.device
    assert output.dtype == q.dtype
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_paged_attention_decode_follows_block_tables_with_gqa_and_empty_context():
    module = load_attention_module()
    torch.manual_seed(11)
    query = torch.randn(3, 4, 3)
    k_cache, v_cache = torch.randn(4, 2, 2, 3), torch.randn(4, 2, 2, 3)
    block_tables = torch.tensor([[2, 0, -1], [3, 1, -1], [-1, -1, -1]])
    context_lens = torch.tensor([3, 4, 0])
    output = module.paged_attention_decode(
        query, k_cache, v_cache, block_tables, context_lens, 0.29, 4, 2, 3, 2
    )
    expected_batches = []
    for batch_index, context_len in enumerate(context_lens.tolist()):
        if context_len == 0:
            expected_batches.append(torch.zeros_like(query[batch_index]))
            continue
        token_indices = torch.arange(context_len)
        physical_blocks = block_tables[batch_index, token_indices // 2]
        offsets = token_indices % 2
        expected_batches.append(
            reference_attention(
                query[batch_index].unsqueeze(0),
                k_cache[physical_blocks, offsets],
                v_cache[physical_blocks, offsets],
                0.29,
                causal=False,
            ).squeeze(0)
        )
    expected = torch.stack(expected_batches)
    assert output.device == query.device
    assert output.dtype == query.dtype
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_mps_cache_and_prefill_preserve_device_and_values():
    module = load_attention_module()
    key_cpu = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    value_cpu = key_cpu + 100
    key, value = key_cpu.to("mps"), value_cpu.to("mps")
    k_cache = torch.full((3, 2, 2, 3), -1.0, device="mps")
    v_cache = torch.full_like(k_cache, -1.0)
    module.store_kvcache(
        key,
        value,
        k_cache,
        v_cache,
        torch.tensor([0, 3, -1, 5], device="mps"),
        block_size=2,
    )
    assert k_cache[0, 0].cpu().equal(key_cpu[0])
    assert k_cache[1, 1].cpu().equal(key_cpu[1])
    assert k_cache[2, 1].cpu().equal(key_cpu[3])

    torch.manual_seed(17)
    q_cpu, k_cpu, v_cpu = (
        torch.randn(4, 4, 3),
        torch.randn(4, 2, 3),
        torch.randn(4, 2, 3),
    )
    output = module.flash_attention_prefill(
        q_cpu.to("mps"),
        k_cpu.to("mps"),
        v_cpu.to("mps"),
        torch.tensor([0, 4], device="mps"),
        0.41,
        4,
        2,
        3,
    )
    expected = reference_attention(q_cpu, k_cpu, v_cpu, 0.41, causal=True)
    assert output.device.type == "mps"
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run the import test to verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_attention.py::test_attention_module_imports_without_triton -v
```

Expected: FAIL while importing `triton`, demonstrating the Mac-incompatible dependency.

- [ ] **Step 3: Keep the tests uncommitted until the first implementation turns them green**

Confirm `git status --short` shows the new test file and the user's existing `attention.py` modification, with no unrelated files staged.

### Task 2: Add DLPack adapters and MLX KV-cache storage

**Files:**
- Modify: `src/myvllm/layers/attention.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Replace Triton imports/kernels with Torch/MLX helpers**

Use these private adapters at the top of `attention.py`:

```python
import mlx.core as mx
import torch
import torch.nn as nn

from myvllm.utils import get_context


def _torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    tensor = tensor.detach().contiguous()
    if tensor.device.type == "mps":
        torch.mps.synchronize()
    return mx.from_dlpack(tensor)


def _mlx_to_torch(array: mx.array, like: torch.Tensor) -> torch.Tensor:
    mx.eval(array)
    result = torch.from_dlpack(array)
    if result.device != like.device or result.dtype != like.dtype:
        result = result.to(device=like.device, dtype=like.dtype)
    return result


def _copy_mlx_to_torch(array: mx.array, destination: torch.Tensor) -> None:
    result = _mlx_to_torch(array, destination)
    if result.data_ptr() != destination.data_ptr():
        destination.copy_(result)
```

- [ ] **Step 2: Implement MLX indexed cache updates**

Retain the public signature and assertions. Filter ignored slots with Torch because MLX 0.32 does not support boolean advanced indices, then do block arithmetic and indexed assignment in MLX:

```python
valid = slot_mapping != -1
if not bool(valid.any().item()):
    return

valid_slots = slot_mapping[valid].to(dtype=torch.long)
slots = _torch_to_mlx(valid_slots).astype(mx.int32)
block_indices = slots // block_size
block_offsets = slots % block_size
k_cache_mx = _torch_to_mlx(k_cache)
v_cache_mx = _torch_to_mlx(v_cache)
k_cache_mx[block_indices, block_offsets] = _torch_to_mlx(key[valid])
v_cache_mx[block_indices, block_offsets] = _torch_to_mlx(value[valid])
mx.eval(k_cache_mx, v_cache_mx)
_copy_mlx_to_torch(k_cache_mx, k_cache)
_copy_mlx_to_torch(v_cache_mx, v_cache)
```

- [ ] **Step 3: Run import and cache tests to verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_attention.py -k "imports_without_triton or store_kvcache" -v
```

Expected: 2 passed.

- [ ] **Step 4: Commit the first green slice**

```bash
git add tests/test_attention.py src/myvllm/layers/attention.py
git commit -m "refactor: move KV cache storage to MLX"
```

### Task 3: Implement MLX variable-length causal prefill

**Files:**
- Modify: `src/myvllm/layers/attention.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Run the prefill test to verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_attention.py::test_flash_attention_prefill_handles_varlen_causal_gqa -v
```

Expected: FAIL until `flash_attention_prefill` uses the new MLX implementation.

- [ ] **Step 2: Implement packed per-sequence MLX attention**

Convert Q/K/V once, preserve packed sequence boundaries, and use native GQA without repeating K/V heads:

```python
q_mx, k_mx, v_mx = _torch_to_mlx(q), _torch_to_mlx(k), _torch_to_mlx(v)
boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
outputs = []
for start, end in zip(boundaries[:-1], boundaries[1:]):
    if start == end:
        continue
    query = mx.expand_dims(mx.transpose(q_mx[start:end], (1, 0, 2)), axis=0)
    key = mx.expand_dims(mx.transpose(k_mx[start:end], (1, 0, 2)), axis=0)
    value = mx.expand_dims(mx.transpose(v_mx[start:end], (1, 0, 2)), axis=0)
    sequence_output = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=scale, mask="causal"
    )
    outputs.append(mx.transpose(sequence_output[0], (1, 0, 2)))
if not outputs:
    return torch.empty_like(q)
return _mlx_to_torch(mx.concatenate(outputs, axis=0), q)
```

- [ ] **Step 3: Run the prefill test to verify GREEN**

Run the same focused test. Expected: 1 passed with CPU Torch output matching the reference.

- [ ] **Step 4: Commit the prefill slice**

```bash
git add src/myvllm/layers/attention.py tests/test_attention.py
git commit -m "feat: run varlen prefill attention with MLX"
```

### Task 4: Implement MLX paged decode and preserve `Attention.forward`

**Files:**
- Modify: `src/myvllm/layers/attention.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Run the paged decode test to verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_attention.py::test_paged_attention_decode_follows_block_tables_with_gqa_and_empty_context -v
```

Expected: FAIL until the Triton decode kernel is replaced.

- [ ] **Step 2: Implement MLX paged gathering and decode**

Convert cache tables once and compute each batch row as follows:

```python
query_mx = _torch_to_mlx(query)
k_cache_mx, v_cache_mx = _torch_to_mlx(k_cache), _torch_to_mlx(v_cache)
block_tables_mx = _torch_to_mlx(block_tables).astype(mx.int32)
context_lengths = context_lens.detach().to(device="cpu", dtype=torch.long).tolist()
outputs = []
for batch_index, context_len in enumerate(context_lengths):
    if context_len == 0:
        outputs.append(mx.zeros(query_mx[batch_index].shape, dtype=query_mx.dtype))
        continue
    token_indices = mx.arange(context_len, dtype=mx.int32)
    physical_blocks = block_tables_mx[batch_index, token_indices // block_size]
    offsets = token_indices % block_size
    key = k_cache_mx[physical_blocks, offsets]
    value = v_cache_mx[physical_blocks, offsets]
    batch_query = mx.expand_dims(mx.expand_dims(query_mx[batch_index], axis=0), axis=2)
    key = mx.expand_dims(mx.transpose(key, (1, 0, 2)), axis=0)
    value = mx.expand_dims(mx.transpose(value, (1, 0, 2)), axis=0)
    batch_output = mx.fast.scaled_dot_product_attention(
        batch_query, key, value, scale=scale
    )
    outputs.append(batch_output[0, :, 0])
return _mlx_to_torch(mx.stack(outputs, axis=0), query)
```

Keep `Attention.__init__` and `Attention.forward` signatures and branching unchanged, removing only Triton-specific comments and operations.

- [ ] **Step 3: Run the complete attention behavior suite to verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_attention.py -v
```

Expected: all current attention tests pass.

- [ ] **Step 4: Commit the decode slice**

```bash
git add src/myvllm/layers/attention.py tests/test_attention.py
git commit -m "feat: run paged decode attention with MLX"
```

### Task 5: Make both main programs valid macOS smoke tests

**Files:**
- Modify: `src/myvllm/layers/attention.py`
- Modify: `src/myvllm/models/qwen3.py`
- Test: `tests/test_attention.py`

- [ ] **Step 1: Add a failing executable-source scan**

Append:

```python
def test_scoped_sources_have_no_cuda_or_triton_execution():
    paths = sorted((ROOT / "src" / "myvllm" / "layers").glob("*.py"))
    paths.append(ROOT / "src" / "myvllm" / "models" / "qwen3.py")
    source = "\n".join(path.read_text() for path in paths)
    assert "import triton" not in source
    assert ".cuda(" not in source
    assert "torch.cuda." not in source
```

Run the test and expect it to fail on the existing CUDA main calls.

- [ ] **Step 2: Replace the attention benchmark with a small MPS prefill/decode smoke test**

Replace the old CUDA timing block with:

```python
if __name__ == "__main__":
    from myvllm.utils import set_context

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required for the attention smoke test")

    torch.manual_seed(0)
    device = torch.device("mps")
    layer = Attention(
        num_heads=4, head_dim=16, num_kv_heads=2, block_size=2
    ).to(device)
    layer.k_cache = torch.zeros(3, 2, 2, 16, device=device)
    layer.v_cache = torch.zeros_like(layer.k_cache)

    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 4], device=device),
        slot_mapping=torch.arange(4, device=device),
    )
    with torch.inference_mode():
        prefill_output = layer(
            torch.randn(4, 4, 16, device=device),
            torch.randn(4, 2, 16, device=device),
            torch.randn(4, 2, 16, device=device),
        )

    set_context(
        is_prefill=False,
        slot_mapping=torch.tensor([4], device=device),
        context_lens=torch.tensor([5], device=device),
        block_tables=torch.tensor([[0, 1, 2]], device=device),
    )
    with torch.inference_mode():
        decode_output = layer(
            torch.randn(1, 4, 16, device=device),
            torch.randn(1, 2, 16, device=device),
            torch.randn(1, 2, 16, device=device),
        )

    torch.mps.synchronize()
    print("prefill:", tuple(prefill_output.shape))
    print("decode:", tuple(decode_output.shape))
```

- [ ] **Step 3: Replace the qwen3 CUDA example with a small initialized MPS prefill**

Add `import torch.distributed as dist` explicitly and use this main block:

```python
if __name__ == "__main__":
    from myvllm.utils import set_context

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required for the Qwen3 smoke test")

    created_process_group = False
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29501",
            rank=0,
            world_size=1,
        )
        created_process_group = True

    try:
        device = torch.device("mps")
        model = Qwen3ForCausalLM(
            vocab_size=128,
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            intermediate_size=128,
            num_layers=1,
        ).to(device).eval()
        num_tokens = 8
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, num_tokens], device=device),
        )
        input_ids = torch.randint(0, 128, (num_tokens,), device=device)
        with torch.inference_mode():
            output = model(input_ids)
        torch.mps.synchronize()
        print("qwen3:", tuple(output.shape))
    finally:
        if created_process_group:
            dist.destroy_process_group()
```

- [ ] **Step 4: Verify the source scan and both main programs**

Run outside the restricted sandbox so Metal is visible:

```bash
uv run --no-sync pytest tests/test_attention.py::test_scoped_sources_have_no_cuda_or_triton_execution -v
uv run --no-sync python src/myvllm/layers/attention.py
uv run --no-sync python src/myvllm/models/qwen3.py
```

Expected: source scan passes; attention main prints prefill `(4, 64)` and decode `(1, 64)` shapes; qwen3 main prints `(8, 64)`.

- [ ] **Step 5: Commit the macOS smoke tests**

```bash
git add src/myvllm/layers/attention.py src/myvllm/models/qwen3.py tests/test_attention.py
git commit -m "test: add macOS MLX attention smoke programs"
```

### Task 6: Final verification and scope review

**Files:**
- Verify: `src/myvllm/layers/attention.py`
- Verify: `src/myvllm/models/qwen3.py`
- Verify: `tests/test_attention.py`

- [ ] **Step 1: Run focused tests**

```bash
uv run --no-sync pytest tests/test_attention.py -v
```

Expected: every attention compatibility test passes.

- [ ] **Step 2: Run the full test suite**

```bash
uv run --no-sync pytest -v
```

Expected: zero failures.

- [ ] **Step 3: Re-run requested main programs from a clean Python process**

```bash
uv run --no-sync python src/myvllm/layers/attention.py
uv run --no-sync python src/myvllm/models/qwen3.py
```

Expected: both exit with status 0 and print their documented shapes.

- [ ] **Step 4: Verify source scope and diff quality**

```bash
rg -n -i "import triton|@triton|\.cuda\(|torch\.cuda\." src/myvllm/layers src/myvllm/models/qwen3.py
git diff --check 2cb0f4f..HEAD
git status --short
```

Expected: no executable Triton/CUDA matches; no whitespace errors in the implementation commits; no unrelated working-tree changes.
