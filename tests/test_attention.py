import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ATTENTION_PATH = ROOT / "src" / "myvllm" / "layers" / "attention.py"


def load_attention_module():
    utils = types.ModuleType("myvllm.utils")
    utils.get_context = lambda: None
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
