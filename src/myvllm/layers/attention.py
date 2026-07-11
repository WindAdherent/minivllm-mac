import torch
import torch.nn as nn
import torch.nn.functional as F

from myvllm.utils import get_context


def _repeat_kv(x: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_heads == num_kv_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_heads, dim=1)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    Store key-value pairs into paged cache.

    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens = key.shape[0]

    # Make contiguous if needed
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"

    slot_mapping = slot_mapping.to(device=k_cache.device, dtype=torch.long)
    valid = slot_mapping != -1
    slots = slot_mapping[valid]
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(slots, block_size)

    k_cache[block_indices, block_offsets] = key[valid]
    v_cache[block_indices, block_offsets] = value[valid]


def flash_attention_prefill(
    q: torch.Tensor,            # shape=[104,16,128]
    k: torch.Tensor,            # shape=[104,8,128]
    v: torch.Tensor,            # shape=[104,8,128]
    cu_seqlens: torch.Tensor,   # [0,70,104]
    scale: float,
    num_heads: int,             # 16
    num_kv_heads: int,          # 8
    head_dim: int,              # 128
) -> torch.Tensor:
    """
    PyTorch scaled dot-product attention for variable-length prefill sequences.

    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor

    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    output = torch.empty_like(q)

    sequence_boundaries = cu_seqlens.to(device="cpu", dtype=torch.long).tolist()
    for sequence_start, sequence_end in zip(
        sequence_boundaries[:-1], sequence_boundaries[1:]
    ):
        if sequence_start == sequence_end:
            continue

        query = q[sequence_start:sequence_end].transpose(0, 1).unsqueeze(0)
        key = k[sequence_start:sequence_end].transpose(0, 1).unsqueeze(0)
        value = v[sequence_start:sequence_end].transpose(0, 1).unsqueeze(0)
        key = _repeat_kv(key, num_heads, num_kv_heads)
        value = _repeat_kv(value, num_heads, num_kv_heads)

        sequence_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
        output[sequence_start:sequence_end] = sequence_output.squeeze(0).transpose(
            0, 1
        )

    return output


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    Compute attention in decode mode using paged KV cache.

    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor

    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    query = query.contiguous()
    output = torch.empty_like(query)
    block_tables = block_tables.to(device=k_cache.device, dtype=torch.long)
    context_lengths = context_lens.to(device="cpu", dtype=torch.long).tolist()

    for batch_index, context_len in enumerate(context_lengths):
        if context_len == 0:
            output[batch_index].zero_()
            continue

        num_blocks = (context_len + block_size - 1) // block_size
        physical_blocks = block_tables[batch_index, :num_blocks]
        key = k_cache.index_select(0, physical_blocks).flatten(0, 1)[:context_len]
        value = v_cache.index_select(0, physical_blocks).flatten(0, 1)[:context_len]

        key = _repeat_kv(key.transpose(0, 1).unsqueeze(0), num_heads, num_kv_heads)
        value = _repeat_kv(
            value.transpose(0, 1).unsqueeze(0), num_heads, num_kv_heads
        )
        batch_query = query[batch_index].unsqueeze(0).unsqueeze(2)
        batch_output = F.scaled_dot_product_attention(
            batch_query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        output[batch_index] = batch_output[0, :, 0]

    return output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store current k, v into cache if cache is allocated
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            # Ensure k, v are in the right shape: (num_tokens, num_kv_heads, head_dim)
            if k.dim() == 4:
                # Batched: (B, N, num_kv_heads, head_dim) -> reshape to (B*N, num_kv_heads, head_dim)
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # Already in correct shape (num_tokens, num_kv_heads, head_dim)
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()

            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # Prefill: 预填充阶段使用 PyTorch SDPA
            # Varlen mode: (total_tokens, num_heads, head_dim)
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")

            o = flash_attention_prefill(q, k, v, cu_seqlens, scale,
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            o = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # Example usage
    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    slot_mapping = torch.arange(N).cuda()

    for _ in range(10):  # Warm-up iterations
        _ = layer(q, k, v)

    import time
    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
