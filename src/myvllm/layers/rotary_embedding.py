import torch.nn as nn
import torch 

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Handle both 3D varlen (total_tokens, num_heads, head_dim) and 4D batched (B, seq_len, num_heads, head_dim)
    if x.dim() == 3:
        # Varlen mode: (total_tokens, num_heads, head_dim)
        total_tokens, num_heads, head_dim = x.shape
        # cos, sin shape: (total_tokens, head_dim/2)
        # Expand to (total_tokens, 1, head_dim/2) for broadcasting
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotary embedding
        # x1, x2 shape: (total_tokens, num_heads, head_dim/2)
        # cos, sin shape: (total_tokens, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)
    else:
        # Batched mode: (B, seq_len, num_heads, head_dim)
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # Expand cos and sin to match the batch and head dimensions
        # cos, sin shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Split x into two halves along the head dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply rotary embedding with proper broadcasting
        # x1, x2 shape: (B, seq_len, num_heads, head_dim/2)
        # cos, sin shape: (1, seq_len, 1, head_dim/2)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self, 
        base:int,
        rotary_embedding: int, 
        max_position: int = 2048,
        is_llama3: bool = False,
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        # 控制各个 RoPE 频率下降的速度，也就是通常公式中的 θ_{base}
        self.base = base
        # base 越大，后面的低频维度旋转得越慢

        # 表示每个 Attention head 中，有多少维度应用 RoPE
        self.rotary_embedding = rotary_embedding
        # 注意，为了实现旋转，rotary_embedding 必须是偶数，因此也有可能是前一半维度应用 RoPE，后一半维度不应用 RoPE

        # 表示提前预计算多少个位置的 cos/sin
        self.max_position = max_position
        # 计算 RoPE 基础频率
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))
        # 此处是公式中的 1/θ_{base}^{-2i/d}，其中 d 是 rotary_embedding，i 是维度索引，由于转换为二维应用旋转矩阵，此处的分子步长为 2

        """
        此处是对 llama3.2 的 RoPE 进行特殊处理：
        高频维度基本保持不变；
        低频维度进行较强缩放；
        中间频率平滑过渡。
        """
        if is_llama3:
            # specifically for llama3.2
            import math
            inv_freq = self.inv_freq
            # no smooth if low_freq_factor == high_freq_factor
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        # 生成全部位置顺序编号
        positions = torch.arange(self.max_position).float()
        # 外积计算旋转角度
        # (max_position, rotary_embedding/2)
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        # 提前计算 cos/sin，方便后续直接使用
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # 前面的 inv_freq 计算的每个维度的缩放，freqs 计算的是每个位置的旋转弧长

        # (max_position, rotary_embedding)
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        # 拼接为一维 [所有 cos | 所有 sin] 方便注册 buffer

        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    # tell the position index of the token
    # apply rotary embedding to query and key
    # 只对 query 和 key 应用 RoPE，value 不应用
    # 因为 RoPE 只影响注意力权重的计算，而 value 只是被用于最终基于注意力分数加权得到结果，不需要旋转
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]  # (seq_len, rotary_embedding)
        cos, sin = cos_sin.chunk(2, dim=-1) # 将原本拼接为一维的 cos/sin 拆分为两半，分别对应 cos 和 sin
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


if __name__ == "__main__":
    base = 5
    # how many dimensions to apply rotary embedding
    rotary_dim = 16
    # maximum position that the long context can reach
    max_position = 100
    print(torch.arange(0, rotary_dim, 2))
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    print(inv_freq)

    t = torch.arange(max_position).float()

    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    print(freqs.size())

    print(freqs[2])
