import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (Sigmoid Linear Unit) activation
    function followed by element-wise multiplication with the input tensor.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量在最后一个维度上分成两部分
        x, y = x.chunk(2, -1)
        # 门控激活：前一半应用SiLU激活函数，后一半保持不变，然后进行元素级乘法
        return F.silu(x) * y

if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().to("mps")
    input_tensor = torch.randn(8, 4000, 8000).to("mps")  # Example input tensor with shape (400, 800)

    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        torch.mps.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.mps.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")