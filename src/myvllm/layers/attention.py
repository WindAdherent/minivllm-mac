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


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
):
    """Store key-value pairs in the caller-owned paged cache."""
    num_tokens = key.shape[0]
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, (
        "Slot mapping size must match number of tokens"
    )

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


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute packed variable-length causal attention with native MLX."""
    q_mx = _torch_to_mlx(q)
    k_mx = _torch_to_mlx(k)
    v_mx = _torch_to_mlx(v)
    boundaries = cu_seqlens.detach().to(
        device="cpu", dtype=torch.long
    ).tolist()
    outputs = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start == end:
            continue
        query = mx.expand_dims(
            mx.transpose(q_mx[start:end], (1, 0, 2)), axis=0
        )
        key = mx.expand_dims(
            mx.transpose(k_mx[start:end], (1, 0, 2)), axis=0
        )
        value = mx.expand_dims(
            mx.transpose(v_mx[start:end], (1, 0, 2)), axis=0
        )
        sequence_output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            mask="causal",
        )
        outputs.append(mx.transpose(sequence_output[0], (1, 0, 2)))

    if not outputs:
        return torch.empty_like(q)
    return _mlx_to_torch(mx.concatenate(outputs, axis=0), q)


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
    block_size: int,
) -> torch.Tensor:
    """Compute decode attention over a paged KV cache with native MLX."""
    query_mx = _torch_to_mlx(query)
    k_cache_mx = _torch_to_mlx(k_cache)
    v_cache_mx = _torch_to_mlx(v_cache)
    block_tables_mx = _torch_to_mlx(block_tables).astype(mx.int32)
    context_lengths = context_lens.detach().to(
        device="cpu", dtype=torch.long
    ).tolist()
    outputs = []

    for batch_index, context_len in enumerate(context_lengths):
        if context_len == 0:
            outputs.append(
                mx.zeros(
                    query_mx[batch_index].shape,
                    dtype=query_mx.dtype,
                )
            )
            continue

        token_indices = mx.arange(context_len, dtype=mx.int32)
        physical_blocks = block_tables_mx[
            batch_index, token_indices // block_size
        ]
        block_offsets = token_indices % block_size
        key = k_cache_mx[physical_blocks, block_offsets]
        value = v_cache_mx[physical_blocks, block_offsets]
        batch_query = mx.expand_dims(
            mx.expand_dims(query_mx[batch_index], axis=0), axis=2
        )
        key = mx.expand_dims(mx.transpose(key, (1, 0, 2)), axis=0)
        value = mx.expand_dims(mx.transpose(value, (1, 0, 2)), axis=0)
        batch_output = mx.fast.scaled_dot_product_attention(
            batch_query,
            key,
            value,
            scale=scale,
        )
        outputs.append(batch_output[0, :, 0])

    if not outputs:
        return torch.empty_like(query)
    return _mlx_to_torch(mx.stack(outputs, axis=0), query)


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
        self.num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else num_heads
        )
        self.block_size = block_size
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if (
            k_cache.numel() > 0
            and v_cache.numel() > 0
            and context.slot_mapping is not None
        ):
            if k.dim() == 4:
                batch_size, sequence_length, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(
                    batch_size * sequence_length,
                    num_kv_heads,
                    head_dim,
                ).contiguous()
                v_to_store = v.reshape(
                    batch_size * sequence_length,
                    num_kv_heads,
                    head_dim,
                ).contiguous()
            else:
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()

            store_kvcache(
                k_to_store,
                v_to_store,
                k_cache,
                v_cache,
                context.slot_mapping,
                self.block_size,
            )

        scale = self.scale / (self.head_dim**0.5)

        if context.is_prefill:
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError(
                    "cu_seqlens_q must be provided for varlen attention"
                )
            output = flash_attention_prefill(
                q,
                k,
                v,
                cu_seqlens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        else:
            output = paged_attention_decode(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        return output.reshape(output.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    layer = Attention(num_heads=8, head_dim=64).cuda()
    batch_size, sequence_length, hidden_size = 4, 1024, 512
    q = torch.randn(batch_size, sequence_length, hidden_size).cuda()
    k = torch.randn(batch_size, sequence_length, hidden_size).cuda()
    v = torch.randn(batch_size, sequence_length, hidden_size).cuda()
    layer.k_cache = torch.zeros(
        batch_size, sequence_length, hidden_size
    ).cuda()
    layer.v_cache = torch.zeros(
        batch_size, sequence_length, hidden_size
    ).cuda()

    for _ in range(10):
        _ = layer(q, k, v)

    import time

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(q, k, v)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    average_time = sum(times) / len(times)
    print(f"Average inference time: {average_time * 1000:.4f} ms")
