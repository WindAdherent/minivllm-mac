import torch.nn as nn 
import torch
# dist 用于分布式训练，提供了进程间通信的功能
import torch.distributed as dist

class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism

        # tp_dim 决定了沿哪一维进行切分
        # y = x @ W.T + b
        # W: [out_features, in_features]
        # tp_dim=0 沿输出维切分，W 切成 [out_features//tp_size, in_features]，每个 GPU 计算自己的输出部分
        # 由于最终参与运算的是 W 的转置，所以实际上是“列并行“
        self.tp_dim = tp_dim 
        # tp_rank 获取当前并行设备的编号
        self.tp_rank = dist.get_rank()
        # tp_size 获取所有并行设备数量
        self.tp_size = dist.get_world_size()
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
class ReplicatedLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # param：当前 GPU 上已经创建好的局部参数，等待被填充
        param_data = param.data 
        # full_dim on the output column
        # 获取 output 维度的大小，即原始未切分的未转置权重矩阵的行数
        full_data_output_size = loaded_weights.size(0)
        # dim size after sharding
        # 按 tp_size 切分后的维度大小，即每个 GPU 负责的输出维度
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        # 计算当前 GPU 负责的输出维度在原始权重矩阵中的起始行索引
        # 类似 cuda kernel 中的 blockIdx.x * blockDim.x
        start_index = self.tp_rank * shard_size
        # 按照第 0 维（行）切分原始权重矩阵，获取当前 GPU 负责的权重子矩阵
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        # 填充当前 GPU 上的参数数据
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# an extension of ColumnParallelLinear by merging several matrices
# 继承于 ColumnParallelLinear，合并多个线性层为一个线性层可以减少计算开销，形式上仅仅是将多个矩阵（如 QKV）合并成一个大矩阵进行计算
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        # output_sizes 是预计合并的多个矩阵的输出维度列表，例如 QKV 合并成一个大矩阵，output_sizes 可能是 [4096, 4096, 4096]
        output_sizes: list[int], # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        # 调用父类的构造函数，使用 output_sizes 的总和作为输出维度，父类会根据 tp_size 进行切分
        super().__init__(input_size, sum(output_sizes), bias)

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to 
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """

        """
        e.g. for QKV merged case:
        input_size = 4096
        output_sizes = [4096, 4096, 4096]
        tp_size = 4
        sum(output_sizes) / tp_size = 12288 / 4 = 3072
        param.shape = [3072, 4096]

        param_data:
        [
            Q_local,   # [1024, 4096]
            K_local,   # [1024, 4096]
            V_local,   # [1024, 4096]
        ]
        """
        param_data = param.data
        # compute offset 
        # 当前 loaded_weights 应该被放到合并参数 param_data 的哪个位置
        # 对于 QKV 合并的情况，假设 loaded_weight_id=1（K），则 offset 就是 Q 在当前设备被切分的维度大小
        # 即 output_sizes[0]//tp_size，表示 K 的权重应该放在 param_data 的 [1024:2048, :] 位置
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # compute size
        # 计算当前加载的权重被切分后的维度大小
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        # 找到当前 rank 上对应 load_weight_id 应该存放的位置，可以想象为 QKV 分别对应的槽位
        # 这句之后 param_data 不再是整个本地合并参数，而是指向其中一段的 view
        # 比如指向当前 rank 上 K 的权重子矩阵，形状从 [3072, 4096] 变为 [1024, 4096]
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        # 从 checkpoint 的当前完整权重 loaded_weights 中，当前 rank 应该取哪一段
        loaded_weights_start_index = self.tp_rank * shard_size
        # 从全局权重 loaded_weights 中切出当前 rank 对应的子矩阵
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        # 利用 param_data 的 view，将切出的权重 shard_weights 复制到当前 rank 的本地参数中
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,                                 # 输入维度 12
        head_size: int,                                  # 头维度 6
        num_heads: int,                                  # q 头 2
        num_kv_heads: int | None = None,                 # kv 头 2
        bias: bool = False,                              # 偏置
    ):
        self.tp_size = dist.get_world_size()             # 全局并行 GPU 数量 2
        num_kv_heads = num_kv_heads or num_heads         # kv头赋值 2
        self.head_size = head_size                       # 头维度赋值 6
        self.num_heads = num_heads // self.tp_size       # 计算每个 GPU 需要维护的 q 头数量 2
        self.num_kv_heads = num_kv_heads // self.tp_size # 计算每个 GPU 需要维护的 kv 头数量 2
        # Calculate per-GPU output size
        # 计算每个 GPU 的输出维度 6 * (2 + 2 * 1) = 24
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Pass TOTAL output size to parent (it will divide by tp_size)
        # 计算整个 qkv 线性层合并后的输出维度 6 * (2 + 2 * 2) = 48
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias) # (12, 48, bias)

    # load_weight_id: q, k, v
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        # batch_size * num_heads * num_token * head_size
        # 初始化线性层参数 (12, 24)
        param_data = param.data
        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        # load_weight_id 必须是 'q', 'k', 'v' 中的一个
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # compute offset
        # 计算偏移
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            # 6 * 2 + 6 * 1 = 18
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        # 本地权重的起始索引
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)

"""
q_proj / k_proj / v_proj：ColumnParallelLinear，切输出
o_proj：RowParallelLinear，切输入

gate_proj / up_proj：ColumnParallelLinear，切输出
down_proj：RowParallelLinear，切输入
"""
# 行并行/输入维度切分，一般用在 Attention 里的 o_proj，MLP 里的 down_proj
class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        # 对输入维度进行切分，输出维度保持不变，tp_dim=1 表示沿输入维切分
        # 由于实际上是 x @ W.T + b，W 的转置会将输入维和输出维交换，所以切分输入维实际上是“行并行“
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the input row
        # 获取输入维度的完整大小，即原始未切分的未转置权重矩阵的列数
        full_data_input_size = loaded_weights.size(1)
        # dim size after sharding
        # 每个 GPU 需要维护的输入维度大小
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # starting index
        # 本地权重切分的起始索引
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每张卡都会得到一个 [N, output_size] 的结果，但这个结果只是完整输出的一部分贡献
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            # 将所有 GPU 上的 result 进行规约求和，并且把求和后的完整结果发回每个 rank
            dist.all_reduce(result, op=dist.ReduceOp.SUM) 
        return result

"""
Tensor Parallel 中，前面的 ColumnParallelLinear 会沿输出维度切分：
    每个 rank 只得到一部分中间激活 x_i，而不是完整的 x。

RowParallelLinear 正好沿输入维度切分权重：
    每个 rank 使用自己的 x_i 和 W_i 计算完整输出的一部分贡献：
        y_i = x_i @ W_i.T

完整输出并不是把 y_i 拼接起来，而是把所有 rank 的贡献相加：
        y = sum_i y_i

因此这里使用 all_reduce(SUM) 来合并各个 rank 的 partial output。
这样可以避免在 ColumnParallelLinear 后立刻 all_gather 中间激活，
让切分后的中间激活直接流入 RowParallelLinear，减少一次通信和中间显存占用。
"""

if __name__ == "__main__":
    # Example usage
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = LinearBase(input_size=10, output_size=5)
    print("LinearBase layer initialized:", layer)