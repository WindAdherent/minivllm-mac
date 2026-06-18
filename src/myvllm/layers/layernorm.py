import torch
import time 

class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # Use nn.Parameter to make gamma learnable and loadable from checkpoints
        # gamma 是一个可学习的缩放参数
        # detach() 切断 gamma 原本可能存在的计算图
        # clone() 创建 gamma 的一个副本，确保它是一个独立的张量，避免影响原始 gamma
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        # eps 是一个小常数，用于数值稳定性，防止除以零
        self.eps = eps

    # 此类中 gamma 实际作为 layer.weight 使用
    # 通过 @property 装饰器提供 layer.gamma 的访问方式，兼容旧代码
    @property
    def gamma(self):
        """Backward compatibility: gamma alias for weight"""
        return self.weight

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ

        # 实际上应该是 mean square 均方值
        # keepdim=True 保持维度不变，方便后续的 x 广播相除
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        # 这里的 gamma 即 self.weight 也广播到 x 的形状进行元素级乘法实现缩放
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual
        # 一份拿去做 RMSNorm 进入下一计算模块，一份原样保存作为下一次的 residual
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # 残差是主干道，为了保持浅层嵌入的特征，每层的输出实际上是输入和残差的和进行 RMSNorm 计算
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            # 注意：如果没有残差输入，直接进行 RMSNorm 计算，适用于初次输入或不需要残差连接的情况
            return self.rms_forward(x)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(8,4000,8000).to("mps")
    gamma = torch.full((8000,), 0.5, device="mps", dtype=x.dtype)
    layer = LayerNorm(gamma=gamma).to("mps")
    residual = torch.full_like(x,fill_value=1)

    for _ in range(10): # Warm-up iterations
        _ = layer(x)
    
    # Without residuals
    times = [] 
    for _ in range(100): # Timing iterations
        torch.mps.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.mps.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # With residuals
    times.clear()
    for _ in range(100): # Timing iterations
        torch.mps.synchronize()
        start_time = time.time()
        _ = layer(x,residual)
        torch.mps.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    