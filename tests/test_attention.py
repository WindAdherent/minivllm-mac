import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ATTENTION_PATH = ROOT / "src" / "myvllm" / "layers" / "attention.py"


def load_attention_module(get_context=lambda: None):
    utils = types.ModuleType("myvllm.utils")
    utils.get_context = get_context
    spec = importlib.util.spec_from_file_location(
        "attention_under_test", ATTENTION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    previous_utils = sys.modules.get("myvllm.utils")
    sys.modules["myvllm.utils"] = utils
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_utils is None:
            del sys.modules["myvllm.utils"]
        else:
            sys.modules["myvllm.utils"] = previous_utils

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


def fail_if_mlx_is_used_for_cpu(_tensor):
    raise AssertionError("CPU fallback must not initialize MLX/Metal")


def test_attention_module_imports_without_triton():
    mlx_was_loaded = "mlx.core" in sys.modules
    module = load_attention_module()

    assert module.Attention is not None
    if not mlx_was_loaded:
        assert "mlx.core" not in sys.modules


def test_store_kvcache_maps_slots_and_skips_negative_entries():
    module = load_attention_module()
    module._torch_to_mlx = fail_if_mlx_is_used_for_cpu
    key = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    value = key + 100
    k_cache = torch.full((3, 2, 2, 3), -1.0)
    v_cache = torch.full_like(k_cache, -1.0)
    k_identity, v_identity = id(k_cache), id(v_cache)

    module.store_kvcache(
        key,
        value,
        k_cache,
        v_cache,
        torch.tensor([0, 3, -1, 5]),
        block_size=2,
    )

    expected_k = torch.full_like(k_cache, -1.0)
    expected_v = torch.full_like(v_cache, -1.0)
    expected_k[0, 0] = key[0]
    expected_k[1, 1] = key[1]
    expected_k[2, 1] = key[3]
    expected_v[0, 0] = value[0]
    expected_v[1, 1] = value[1]
    expected_v[2, 1] = value[3]

    assert id(k_cache) == k_identity
    assert id(v_cache) == v_identity
    torch.testing.assert_close(k_cache, expected_k)
    torch.testing.assert_close(v_cache, expected_v)


def test_flash_attention_prefill_handles_varlen_causal_gqa():
    module = load_attention_module()
    module._torch_to_mlx = fail_if_mlx_is_used_for_cpu
    torch.manual_seed(7)
    q = torch.randn(5, 4, 3)
    k = torch.randn(5, 2, 3)
    v = torch.randn(5, 2, 3)

    output = module.flash_attention_prefill(
        q,
        k,
        v,
        torch.tensor([0, 2, 5]),
        0.37,
        num_heads=4,
        num_kv_heads=2,
        head_dim=3,
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
    module._torch_to_mlx = fail_if_mlx_is_used_for_cpu
    torch.manual_seed(11)
    query = torch.randn(3, 4, 3)
    k_cache = torch.randn(4, 2, 2, 3)
    v_cache = torch.randn(4, 2, 2, 3)
    block_tables = torch.tensor(
        [[2, 0, -1], [3, 1, -1], [-1, -1, -1]]
    )
    context_lens = torch.tensor([3, 4, 0])

    output = module.paged_attention_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        0.29,
        num_heads=4,
        num_kv_heads=2,
        head_dim=3,
        block_size=2,
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
def test_paged_attention_decode_skips_unmapped_cache_blocks():
    module = load_attention_module()
    torch.manual_seed(13)
    query_cpu = torch.randn(1, 4, 3)
    k_cache_cpu = torch.randn(3, 2, 2, 3)
    v_cache_cpu = torch.randn(3, 2, 2, 3)

    output = module.paged_attention_decode(
        query_cpu.to("mps"),
        k_cache_cpu.to("mps"),
        v_cache_cpu.to("mps"),
        torch.tensor([[1, -1]], device="mps"),
        torch.tensor([3], device="mps"),
        0.31,
        num_heads=4,
        num_kv_heads=2,
        head_dim=3,
        block_size=2,
    )

    expected = reference_attention(
        query_cpu,
        k_cache_cpu[1],
        v_cache_cpu[1],
        0.31,
        causal=False,
    )
    torch.testing.assert_close(
        output.cpu(), expected, rtol=1e-5, atol=1e-5
    )


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
    q_cpu = torch.randn(4, 4, 3)
    k_cpu = torch.randn(4, 2, 3)
    v_cpu = torch.randn(4, 2, 3)
    output = module.flash_attention_prefill(
        q_cpu.to("mps"),
        k_cpu.to("mps"),
        v_cpu.to("mps"),
        torch.tensor([0, 4], device="mps"),
        0.41,
        num_heads=4,
        num_kv_heads=2,
        head_dim=3,
    )
    expected = reference_attention(q_cpu, k_cpu, v_cpu, 0.41, causal=True)

    assert output.device.type == "mps"
    torch.testing.assert_close(
        output.cpu(), expected, rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_attention_forward_runs_mps_prefill_and_decode_end_to_end():
    current_context = [None]
    module = load_attention_module(lambda: current_context[0])
    layer = module.Attention(
        num_heads=4,
        head_dim=3,
        num_kv_heads=2,
        block_size=2,
    ).to("mps")
    layer.k_cache = torch.zeros(3, 2, 2, 3, device="mps")
    layer.v_cache = torch.zeros_like(layer.k_cache)

    torch.manual_seed(19)
    q_cpu = torch.randn(4, 4, 3)
    k_cpu = torch.randn(4, 2, 3)
    v_cpu = torch.randn(4, 2, 3)
    current_context[0] = types.SimpleNamespace(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 4], device="mps"),
        slot_mapping=torch.arange(4, device="mps"),
        context_lens=None,
        block_tables=None,
    )

    prefill_output = layer(
        q_cpu.to("mps"), k_cpu.to("mps"), v_cpu.to("mps")
    )
    expected_prefill = reference_attention(
        q_cpu, k_cpu, v_cpu, 3**-0.5, causal=True
    ).reshape(4, 12)

    assert prefill_output.shape == (4, 12)
    assert prefill_output.device.type == "mps"
    assert prefill_output.dtype == q_cpu.dtype
    torch.testing.assert_close(
        prefill_output.cpu(), expected_prefill, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        layer.k_cache.flatten(0, 1)[:4].cpu(), k_cpu
    )

    decode_q_cpu = torch.randn(1, 4, 3)
    decode_k_cpu = torch.randn(1, 2, 3)
    decode_v_cpu = torch.randn(1, 2, 3)
    current_context[0] = types.SimpleNamespace(
        is_prefill=False,
        cu_seqlens_q=None,
        slot_mapping=torch.tensor([4], device="mps"),
        context_lens=torch.tensor([5], device="mps"),
        block_tables=torch.tensor([[0, 1, 2]], device="mps"),
    )

    decode_output = layer(
        decode_q_cpu.to("mps"),
        decode_k_cpu.to("mps"),
        decode_v_cpu.to("mps"),
    )
    expected_decode = reference_attention(
        decode_q_cpu,
        torch.cat([k_cpu, decode_k_cpu]),
        torch.cat([v_cpu, decode_v_cpu]),
        3**-0.5,
        causal=False,
    ).reshape(1, 12)

    assert decode_output.shape == (1, 12)
    assert decode_output.device.type == "mps"
    assert decode_output.dtype == decode_q_cpu.dtype
    torch.testing.assert_close(
        decode_output.cpu(), expected_decode, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(layer.k_cache[2, 0].cpu(), decode_k_cpu[0])


def test_scoped_sources_have_no_cuda_or_triton_execution():
    paths = sorted((ROOT / "src" / "myvllm" / "layers").glob("*.py"))
    qwen_path = ROOT / "src" / "myvllm" / "models" / "qwen3.py"
    paths.append(qwen_path)
    source = "\n".join(path.read_text() for path in paths)

    assert "import triton" not in source
    assert ".cuda(" not in source
    assert "torch.cuda." not in source
    assert "tcp://127.0.0.1:29501" not in qwen_path.read_text()
