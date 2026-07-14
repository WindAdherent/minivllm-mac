# ModelRunner on macOS MPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CUDA/NCCL execution in `ModelRunner` with single-process PyTorch MPS execution while preserving its Torch-facing inference behavior.

**Architecture:** Keep `ModelRunner`, model classes, sequence/context data, sampler, and KV-cache layout in PyTorch. Centralize MPS validation in a small helper, place all runtime tensors on one `torch.device('mps')`, account for cache capacity with MPS working-set APIs, and run prefill/decode eagerly because CUDA Graph replay has no MPS equivalent in scope. The existing attention layer may continue using MLX internally without exposing MLX values through `ModelRunner`.

**Tech Stack:** Python 3.11, PyTorch 2.12 MPS, MLX 0.32 through the existing attention layer, Gloo world-size one, pytest 9.1, uv.

---

## File map and workspace constraints

- Create `tests/test_model_runner.py`: isolated regression tests for platform validation, MPS tensor preparation, cache sizing, eager decode, cleanup, and removal of CUDA/NCCL execution.
- Modify `src/myvllm/engine/model_runner.py`: the only production file changed; retain model selection, checkpoint loading, sequence/context behavior, sampling, and shared-memory methods.
- Update `docs/superpowers/plans/2026-07-13-model-runner-mps.md`: check off completed steps during execution.

`src/myvllm/engine/model_runner.py` began as a 458-line user-owned worktree modification relative to an empty tracked file. Its content was copied by patch into the isolated feature worktree before implementation. Do not reset, restore, or replace it wholesale; apply focused patches to the copied content and review its complete final diff before staging.

## Execution status — implementation verified 2026-07-14

Implementation and task-level reviews are complete in the isolated worktree at `/Users/chowhound/VSCodeProjects/minivllm-mac/.worktrees/model-runner-mps` on branch `feat/model-runner-mps`:

- Task 1 is complete and reviewed at `0278337`.
  - Runtime validation, single-rank Gloo initialization, and MPS model/default-device placement are retained.
- Task 2 is complete and reviewed across `06344a5`, `9790baa`, and `51e9a3f`.
  - Prefill, decode, context, block-table, and sampling tensors use MPS, with real-`Sequence`, prefix-cache, dtype, and device coverage.
- Task 3 was implemented at `ba2af91` and its warmup-memory reserve was fixed at `bbb6d01`.
  - MPS warmup APIs and single-rank working-set cache sizing are implemented; specification and code-quality reviews passed.
  - A possible refinement to reduce overlap in the deliberately conservative warmup reserve remains a nonblocking suggestion, not a correctness or approval blocker.
- Task 4 is complete and reviewed at `5c82183`.
  - Decode runs eagerly on MPS and the remaining graph-capture lifecycle was removed without changing sampling or context-reset behavior.
- Task 5 fresh verification completed on 2026-07-14:
  - `/Users/chowhound/VSCodeProjects/minivllm-mac/.venv/bin/python -m pytest -q` exited 0 with `23 passed in 2.02s`, zero failures, and no MPS skips.
  - `/Users/chowhound/VSCodeProjects/minivllm-mac/.venv/bin/python -m py_compile src/myvllm/engine/model_runner.py tests/test_model_runner.py` exited 0 with no output.
  - Case-insensitive `cuda|nccl` and graph-state scans (`capture|replay|graph_pool|graph_vars|self\.graphs`) each exited 1 with no matches in `model_runner.py`.
  - `git diff --check` exited 0, and `git status --short` was clean before this status update.
  - Relative to `d94844c`, the feature changes are limited to `src/myvllm/engine/model_runner.py`, `tests/test_model_runner.py`, and this implementation plan; no file outside the allowed runner, tests, design-spec, and implementation-plan scope changed.
  - The main worktree's pre-existing modification to `src/myvllm/engine/model_runner.py` is user-owned and was not changed while executing the feature worktree.

Final whole-feature review remains next. The branch has not been integrated or merged.

### Task 1: Establish isolated tests and MPS runtime initialization

**Files:**
- Create: `tests/test_model_runner.py`
- Modify: `src/myvllm/engine/model_runner.py:1-102`

- [ ] **Step 1: Create an isolated module loader and write failing runtime-validation tests**

Create `tests/test_model_runner.py` with stubs for currently incomplete neighboring modules. This keeps the tests scoped to `model_runner.py` rather than requiring unrelated sampler, Llama, or checkpoint-loader implementation.

```python
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "src" / "myvllm" / "engine" / "model_runner.py"


class DummyModel(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.moved_to = None

    def to(self, device):
        self.moved_to = torch.device(device)
        return self

    def forward(self, input_ids):
        return input_ids

    def compute_logits(self, hidden_states):
        return hidden_states


class DummySampler:
    def __call__(self, logits, _temperatures):
        return logits.argmax(dim=-1).tolist()


class DummySequence:
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size


def load_model_runner_module(monkeypatch):
    context_state = SimpleNamespace(value=SimpleNamespace())

    def set_context(**kwargs):
        context_state.value = SimpleNamespace(**kwargs)

    def get_context():
        return context_state.value

    def reset_context():
        context_state.value = SimpleNamespace()

    qwen_module = types.ModuleType("myvllm.models.qwen3")
    qwen_module.Qwen3ForCausalLM = DummyModel
    llama_module = types.ModuleType("myvllm.models.llama")
    llama_module.LlamaForCausalLM = DummyModel
    sampler_module = types.ModuleType("myvllm.layers.sampler")
    sampler_module.SamplerLayer = DummySampler
    sequence_module = types.ModuleType("myvllm.engine.sequence")
    sequence_module.Sequence = DummySequence
    utils_module = types.ModuleType("myvllm.utils")
    utils_module.__path__ = []
    utils_module.set_context = set_context
    utils_module.get_context = get_context
    utils_module.reset_context = reset_context
    loader_module = types.ModuleType("myvllm.utils.loader")
    loader_module.load_weights_from_checkpoint = lambda _model, _path: None

    monkeypatch.setitem(sys.modules, "myvllm.models.qwen3", qwen_module)
    monkeypatch.setitem(sys.modules, "myvllm.models.llama", llama_module)
    monkeypatch.setitem(sys.modules, "myvllm.layers.sampler", sampler_module)
    monkeypatch.setitem(sys.modules, "myvllm.engine.sequence", sequence_module)
    monkeypatch.setitem(sys.modules, "myvllm.utils", utils_module)
    monkeypatch.setitem(sys.modules, "myvllm.utils.loader", loader_module)

    spec = importlib.util.spec_from_file_location(
        "model_runner_under_test", MODEL_RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._test_context_state = context_state
    return module


def test_mps_device_rejects_unsupported_runtime(monkeypatch):
    module = load_model_runner_module(monkeypatch)

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="world_size == 1"):
        module._mps_device(world_size=2, rank=0)
    with pytest.raises(ValueError, match="rank == 0"):
        module._mps_device(world_size=1, rank=1)

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        module._mps_device(world_size=1, rank=0)
```

- [ ] **Step 2: Run the validation test and verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py::test_mps_device_rejects_unsupported_runtime -v
```

Expected: FAIL with `AttributeError: module 'model_runner_under_test' has no attribute '_mps_device'`.

- [ ] **Step 3: Add the minimal MPS runtime helper**

Add below the imports in `model_runner.py`:

```python
def _mps_device(world_size: int, rank: int) -> torch.device:
    if world_size != 1:
        raise ValueError("macOS MPS ModelRunner requires world_size == 1")
    if rank != 0:
        raise ValueError("macOS MPS ModelRunner requires rank == 0")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable on this Mac")
    return torch.device("mps")
```

- [ ] **Step 4: Run the validation test and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write a failing constructor-placement test**

Append:

```python
def qwen_config():
    return {
        "block_size": 2,
        "world_size": 1,
        "enforce_eager": True,
        "model_name_or_path": "/tmp/Qwen3-0.6B",
        "vocab_size": 32,
        "hidden_size": 8,
        "num_heads": 2,
        "head_dim": 4,
        "scale": 0.5,
        "num_kv_heads": 2,
        "rms_norm_epsilon": 1e-5,
        "qkv_bias": False,
        "base": 10000,
        "max_position": 32,
        "intermediate_size": 16,
        "ffn_bias": False,
        "num_layers": 1,
        "tie_word_embeddings": False,
    }


def test_constructor_uses_gloo_and_moves_model_to_mps(monkeypatch):
    module = load_model_runner_module(monkeypatch)
    init_calls = []
    default_devices = []

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        module.dist,
        "init_process_group",
        lambda backend, init_method, world_size, rank: init_calls.append(
            (backend, init_method, world_size, rank)
        ),
    )
    monkeypatch.setattr(module.torch, "set_default_device", default_devices.append)
    monkeypatch.setattr(module.ModelRunner, "warmup_model", lambda _self: None)
    monkeypatch.setattr(module.ModelRunner, "allocate_kv_cache", lambda _self: None)

    runner = module.ModelRunner(qwen_config(), rank=0, event=None)

    assert runner.device == torch.device("mps")
    assert runner.model.moved_to == torch.device("mps")
    assert init_calls == [("gloo", "tcp://localhost:12345", 1, 0)]
    assert default_devices == [torch.device("mps")]
```

- [ ] **Step 6: Run the constructor test and verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py::test_constructor_uses_gloo_and_moves_model_to_mps -v
```

Expected: FAIL because construction still initializes NCCL or calls a CUDA device API.

- [ ] **Step 7: Switch initialization and model placement to MPS**

In `ModelRunner.__init__`, retain the existing assignments and model-selection match, but replace the rank/process/device setup and model move with:

```python
self.rank = rank
self.device = _mps_device(self.world_size, self.rank)
dist.init_process_group(
    "gloo",
    "tcp://localhost:12345",
    world_size=self.world_size,
    rank=self.rank,
)

# existing model-selection match remains unchanged

self.model = self.model.to(self.device)
```

Replace the default device assignment at the end of initialization with:

```python
torch.set_default_device(self.device)
torch.set_default_dtype(self.default_dtype)
```

Leave graph-capture removal for Task 4 so its regression test can be observed failing first.

- [ ] **Step 8: Run Task 1 tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py -k "mps_device or constructor" -v
```

Expected: 2 passed.

- [ ] **Step 9: Commit Task 1**

```bash
git add tests/test_model_runner.py
git add src/myvllm/engine/model_runner.py
git commit -m "feat: initialize model runner on MPS"
```

Before staging, verify `git diff -- src/myvllm/engine/model_runner.py` still contains the original user-owned model-runner implementation plus only the scoped MPS edits.

### Task 2: Place prepared inference tensors on MPS

**Files:**
- Modify: `tests/test_model_runner.py`
- Modify: `src/myvllm/engine/model_runner.py:260-347`

- [ ] **Step 1: Add a reusable fake sequence and a failing prefill test**

Append:

```python
class FakeSequence:
    def __init__(
        self,
        token_ids,
        block_table,
        block_size=2,
        num_cached_tokens=0,
        temperature=0.7,
    ):
        self.token_ids = token_ids
        self.last_token = token_ids[-1]
        self.block_table = block_table
        self.block_size = block_size
        self.num_cached_tokens = num_cached_tokens
        self.temperature = temperature

    def __len__(self):
        return len(self.token_ids)

    @property
    def num_cached_blocks(self):
        return (self.num_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def num_blocks(self):
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        remainder = len(self.token_ids) % self.block_size
        return remainder or self.block_size


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_prepare_prefill_builds_mps_context(monkeypatch):
    module = load_model_runner_module(monkeypatch)
    runner = module.ModelRunner.__new__(module.ModelRunner)
    runner.block_size = 2
    runner.device = torch.device("mps")
    seqs = [FakeSequence([10, 11, 12], [2, 3])]

    input_ids = runner.prepare_prefill(seqs)
    context = module._test_context_state.value

    assert input_ids.device.type == "mps"
    assert input_ids.cpu().tolist() == [10, 11, 12]
    assert context.slot_mapping.cpu().tolist() == [4, 5, 6]
    assert context.cu_seqlens_q.dtype == torch.int32
    assert context.cu_seqlens_q.cpu().tolist() == [0, 3]
    assert context.cu_seqlens_k.cpu().tolist() == [0, 3]
    assert context.block_tables is None
```

- [ ] **Step 2: Run the prefill test and verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py::test_prepare_prefill_builds_mps_context -v
```

Expected: FAIL when the existing `.cuda(non_blocking=True)` path runs on the Mac.

- [ ] **Step 3: Move prefill tensors directly to the runner device**

Replace the tensor-construction tail of `prepare_prefill` with:

```python
input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
slot_mapping_tensor = torch.tensor(
    slot_mappings, dtype=torch.long, device=self.device
)

set_context(
    is_prefill=True,
    cu_seqlens_q=torch.tensor(
        cu_seqlens_q, dtype=torch.int32, device=self.device
    ),
    cu_seqlens_k=torch.tensor(
        cu_seqlens_k, dtype=torch.int32, device=self.device
    ),
    max_seqlen_q=max(seqlens_q),
    max_seqlen_k=max(seqlens_k),
    slot_mapping=slot_mapping_tensor,
    context_lens=None,
    block_tables=(
        torch.tensor(block_tables, dtype=torch.int32, device=self.device)
        if block_tables
        else None
    ),
)
return input_ids
```

- [ ] **Step 4: Run the prefill test and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Add failing decode and sampling tests**

Append:

```python
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_prepare_decode_and_sample_build_mps_tensors(monkeypatch):
    module = load_model_runner_module(monkeypatch)
    runner = module.ModelRunner.__new__(module.ModelRunner)
    runner.block_size = 2
    runner.device = torch.device("mps")
    seqs = [
        FakeSequence([10, 11, 12], [2, 3], temperature=0.25),
        FakeSequence([20, 21], [7], temperature=0.9),
    ]

    input_ids = runner.prepare_decode(seqs)
    temperatures = runner.prepare_sample(seqs)
    context = module._test_context_state.value

    assert input_ids.device.type == "mps"
    assert input_ids.cpu().tolist() == [12, 21]
    assert context.slot_mapping.cpu().tolist() == [6, 15]
    assert context.context_lens.cpu().tolist() == [3, 2]
    assert context.block_tables.dtype == torch.int32
    assert context.block_tables.cpu().tolist() == [[2, 3], [7, -1]]
    assert temperatures.device.type == "mps"
    torch.testing.assert_close(
        temperatures.cpu(), torch.tensor([0.25, 0.9], dtype=torch.float32)
    )
```

- [ ] **Step 6: Run the decode test and verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py::test_prepare_decode_and_sample_build_mps_tensors -v
```

Expected: FAIL when a CUDA tensor transfer is attempted.

- [ ] **Step 7: Move decode and sampling tensors directly to MPS**

In `prepare_decode`, replace every pinned CPU tensor plus `.cuda(non_blocking=True)` construction with direct `device=self.device` construction:

```python
input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
set_context(
    is_prefill=False,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=torch.tensor(
        slot_mappings, dtype=torch.long, device=self.device
    ),
    context_lens=torch.tensor(
        context_lens, dtype=torch.long, device=self.device
    ),
    block_tables=(
        torch.tensor(block_tables, dtype=torch.int32, device=self.device)
        if block_tables
        else None
    ),
)
return input_ids
```

Replace `prepare_sample` with:

```python
def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
    return torch.tensor(
        [seq.temperature for seq in seqs],
        dtype=torch.float32,
        device=self.device,
    )
```

- [ ] **Step 8: Run Task 2 tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py -k "prepare_prefill or prepare_decode" -v
```

Expected: 2 passed.

- [ ] **Step 9: Commit Task 2**

```bash
git add tests/test_model_runner.py
git add src/myvllm/engine/model_runner.py
git commit -m "feat: prepare inference tensors on MPS"
```

### Task 3: Replace CUDA warmup and KV-cache accounting with MPS APIs

**Files:**
- Modify: `tests/test_model_runner.py`
- Modify: `src/myvllm/engine/model_runner.py:178-253`

- [ ] **Step 1: Write a failing warmup synchronization test**

Append:

```python
def test_warmup_uses_mps_cache_and_synchronization(monkeypatch):
    module = load_model_runner_module(monkeypatch)
    runner = module.ModelRunner.__new__(module.ModelRunner)
    runner.config = {
        "max_num_batch_tokens": 8,
        "max_model_length": 4,
        "block_size": 2,
    }
    calls = []

    monkeypatch.setattr(module.torch.mps, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(module.torch.mps, "synchronize", lambda: calls.append("sync"))
    runner.run = lambda seqs, is_prefill: calls.append((len(seqs), is_prefill))

    runner.warmup_model()

    assert calls == ["empty", (2, True), "sync", "empty"]
```

- [ ] **Step 2: Run the warmup test and verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py::test_warmup_uses_mps_cache_and_synchronization -v
```

Expected: FAIL because `warmup_model` calls CUDA memory APIs.

- [ ] **Step 3: Implement MPS warmup synchronization**

Replace `warmup_model` with:

```python
def warmup_model(self):
    torch.mps.empty_cache()
    max_tokens = self.config["max_num_batch_tokens"]
    max_model_length = self.config["max_model_length"]
    batch_size = max_tokens // max_model_length
    seqs = [
        Sequence(
            token_ids=[0] * max_model_length,
            block_size=self.config["block_size"],
        )
        for _ in range(batch_size)
    ]
    self.run(seqs, is_prefill=True)
    torch.mps.synchronize()
    torch.mps.empty_cache()
```

- [ ] **Step 4: Run the warmup test and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write a failing deterministic cache-budget test**

Append:

```python
class CacheLayer:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None


class CacheModel:
    def __init__(self, layers):
        self.layers = layers

    def modules(self):
        return iter(self.layers)


def test_allocate_kv_cache_uses_mps_working_set_budget(monkeypatch):
    module = load_model_runner_module(monkeypatch)
    runner = module.ModelRunner.__new__(module.ModelRunner)
    runner.config = {
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 4,
        "gpu_memory_utilization": 0.5,
    }
    runner.world_size = 1
    runner.rank = 0
    runner.block_size = 2
    runner.default_dtype = torch.float32
    runner.device = torch.device("cpu")
    layers = [CacheLayer(), CacheLayer()]
    runner.model = CacheModel(layers)

    monkeypatch.setattr(module.torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(
        module.torch.mps, "recommended_max_memory", lambda: 1024
    )
    monkeypatch.setattr(module.torch.mps, "driver_allocated_memory", lambda: 128)

    runner.allocate_kv_cache()

    assert runner.config["max_cached_blocks"] == 1
    for layer in layers:
        assert layer.k_cache.shape == (1, 2, 2, 4)
        assert layer.v_cache.shape == (1, 2, 2, 4)
        assert torch.count_nonzero(layer.k_cache) == 0
        assert torch.count_nonzero(layer.v_cache) == 0
```

The expected block count follows `(1024 * 0.5 - 128) // 256 == 1`, where one block consumes `2 * 2 * 2 * 2 * 4 * 4 == 256` bytes.

- [ ] **Step 6: Run the cache-budget test and verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py::test_allocate_kv_cache_uses_mps_working_set_budget -v
```

Expected: FAIL when `torch.cuda.mem_get_info()` is reached.

- [ ] **Step 7: Implement MPS cache budgeting and allocation**

Replace the memory-query and distributed-reduction portions of `allocate_kv_cache` with:

```python
torch.mps.synchronize()
memory_limit = int(
    torch.mps.recommended_max_memory()
    * self.config["gpu_memory_utilization"]
)
available_mem = memory_limit - torch.mps.driver_allocated_memory()

num_layers = self.config["num_layers"]
num_kv_heads = self.config["num_kv_heads"]
if "head_dim" in self.config:
    head_dim = self.config["head_dim"]
else:
    head_dim = self.config["hidden_size"] // self.config["num_heads"]
block_bytes = (
    self.block_size
    * 2
    * num_layers
    * num_kv_heads
    * head_dim
    * self.default_dtype.itemsize
)
num_available_kv_blocks = int(available_mem // block_bytes)
assert num_available_kv_blocks >= 1, (
    f"Not enough memory to hold at least one block of KV cache on rank {self.rank}"
)
self.config["max_cached_blocks"] = num_available_kv_blocks
if self.rank == 0:
    print(
        "[Rank 0] Global max_cached_blocks (min): "
        f"{self.config['max_cached_blocks']}"
    )

allocated_kv_cache = torch.zeros(
    2,
    num_layers,
    self.config["max_cached_blocks"],
    self.block_size,
    num_kv_heads,
    head_dim,
    dtype=self.default_dtype,
    device=self.device,
)
```

Retain the existing loop that assigns per-layer `k_cache` and `v_cache` views. Remove the cross-rank reduction because the validated runtime has one rank.

- [ ] **Step 8: Run Task 3 tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py -k "warmup or allocate_kv_cache" -v
```

Expected: 2 passed.

- [ ] **Step 9: Commit Task 3**

```bash
git add tests/test_model_runner.py
git add src/myvllm/engine/model_runner.py
git commit -m "feat: size KV cache for MPS memory"
```

### Task 4: Use eager decode and remove the remaining CUDA graph lifecycle

**Files:**
- Modify: `tests/test_model_runner.py`
- Modify: `src/myvllm/engine/model_runner.py:145-458`

- [ ] **Step 1: Write a failing eager-decode regression test**

Append:

```python
class EagerModel:
    def __call__(self, input_ids):
        return input_ids.float() + 1

    def compute_logits(self, hidden_states):
        return hidden_states * 2


@pytest.mark.parametrize("is_prefill", [True, False])
def test_run_model_is_eager_for_prefill_and_decode(monkeypatch, is_prefill):
    module = load_model_runner_module(monkeypatch)
    runner = module.ModelRunner.__new__(module.ModelRunner)
    runner.model = EagerModel()
    runner.enforce_eager = False
    input_ids = torch.tensor([1, 3])

    logits = runner.run_model(input_ids, is_prefill=is_prefill)

    torch.testing.assert_close(logits, torch.tensor([4.0, 8.0]))
```

- [ ] **Step 2: Run the decode parameter and verify RED**

Run:

```bash
uv run --no-sync pytest 'tests/test_model_runner.py::test_run_model_is_eager_for_prefill_and_decode[False]' -v
```

Expected: FAIL with a missing `graphs` or `graph_vars` attribute, proving decode still requires CUDA Graph state.

- [ ] **Step 3: Make both model paths eager**

Replace `run_model` with:

```python
@torch.inference_mode()
def run_model(
    self, input_ids: torch.Tensor, is_prefill: bool
) -> torch.Tensor:
    hidden_states = self.model(input_ids)
    return self.model.compute_logits(hidden_states)
```

Keep `is_prefill` in the signature for caller compatibility even though MPS uses the same eager path for both values.

- [ ] **Step 4: Run the eager test and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py::test_run_model_is_eager_for_prefill_and_decode -v
```

Expected: 2 passed.

- [ ] **Step 5: Write failing cleanup and source-scan tests**

Append:

```python
def test_exit_synchronizes_mps_and_destroys_process_group(monkeypatch):
    module = load_model_runner_module(monkeypatch)
    runner = module.ModelRunner.__new__(module.ModelRunner)
    runner.world_size = 1
    runner.enforce_eager = True
    calls = []

    monkeypatch.setattr(module.torch.mps, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        module.dist, "destroy_process_group", lambda: calls.append("destroy")
    )

    runner.exit()

    assert calls == ["sync", "destroy"]


def test_model_runner_source_has_no_cuda_or_nccl_execution():
    source = MODEL_RUNNER_PATH.read_text().lower()
    forbidden = ("torch.cuda", ".cuda(", "cuda:", "cudagraph", "nccl")

    for token in forbidden:
        assert token not in source
```

- [ ] **Step 6: Run cleanup and source scan and verify RED**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py -k "exit_synchronizes or no_cuda" -v
```

Expected: both tests FAIL because cleanup and graph capture still use CUDA.

- [ ] **Step 7: Remove CUDA graph initialization, capture, and cleanup**

Make these focused production edits:

```python
# Remove the unused `import math`.

# In __init__, remove this complete block:
# if not self.enforce_eager:
#     self.capture_cudagraph()

# In exit, remove graph deletion and replace device synchronization with:
torch.mps.synchronize()
```

Delete the complete `capture_cudagraph` method, including graph buffers, graph pools, replay capture, and CUDA synchronization. Update nearby comments to describe eager MPS execution and remove obsolete CUDA wording. Do not change `run`, shared-memory methods, sampling conditions, or context reset.

- [ ] **Step 8: Run all focused tests and verify GREEN**

Run:

```bash
uv run --no-sync pytest tests/test_model_runner.py -v
```

Expected: all model-runner tests pass with no failures or warnings from the changed code.

- [ ] **Step 9: Commit Task 4**

```bash
git add tests/test_model_runner.py
git add src/myvllm/engine/model_runner.py
git commit -m "feat: run model decode eagerly on MPS"
```

### Task 5: Verify integration and preserve scope

**Files:**
- Modify only if verification exposes a scoped defect: `tests/test_model_runner.py`
- Modify only if verification exposes a scoped defect: `src/myvllm/engine/model_runner.py`
- Modify: `docs/superpowers/plans/2026-07-13-model-runner-mps.md`

- [ ] **Step 1: Run the complete project test suite with Metal access**

Run:

```bash
uv run --no-sync pytest -q
```

Expected: all tests pass; MPS tests execute on the target Mac rather than being skipped.

- [ ] **Step 2: Compile the changed Python files**

Run:

```bash
uv run --no-sync python -m py_compile src/myvllm/engine/model_runner.py tests/test_model_runner.py
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Scan the requested source file for stale CUDA/NCCL content**

Run:

```bash
rg -n -i "cuda|nccl" src/myvllm/engine/model_runner.py
```

Expected: exit code 1 with no matches.

- [ ] **Step 4: Review formatting and scope**

Run:

```bash
git diff --check
git status --short
git diff -- src/myvllm/engine/model_runner.py tests/test_model_runner.py
```

Expected: no whitespace errors; only the requested model runner, its focused tests, and plan progress are part of this implementation. Confirm the final `model_runner.py` still contains the original model constructors, checkpoint loader call, shared-memory protocol, sequence preparation logic, sampler call, and context reset.

- [ ] **Step 5: Commit final plan progress only if it changed after Task 4**

```bash
git add docs/superpowers/plans/2026-07-13-model-runner-mps.md
git commit -m "docs: record MPS model runner verification"
```

Skip this commit only when the plan file has no execution-progress changes to record.
