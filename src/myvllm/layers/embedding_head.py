import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.utils import get_context


# vocabparallelembedding
# shard over the number of vocab, not the embedding size

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # keep the original num_embeddings
        # 原始词表长度
        self.num_embeddings = num_embeddings
        # pad to make it divisible by tp_size
        # 向上补齐，需要补齐到能被 tp_size 整除的长度
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        # this is the num_embeddings per partition in this current GPU
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim # 嵌入维度

        # 初始化权重参数，每张卡需要维护的词表长度，嵌入维度
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader # 权重加载器

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data # 权重矩阵

        offset = self.tp_rank * self.num_embeddings_per_partition # 本地权重的起始位置
        shard_size = self.num_embeddings_per_partition # 切片大小，即每张卡需要维护的词表长度

        # calculate how much of the original vocab falls in this partition
        # 计算本地权重的实际起始位置和结束位置，确保不会超过原始词表长度
        actual_start = min(offset, self.num_embeddings) # 起点
        actual_end = min(offset + shard_size, self.num_embeddings) # 终点，避免移除
        actual_size = max(0, actual_end - actual_start) # 长度

        if actual_size > 0:
            # load the actual weights
            # 加载权重
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            param_data[:actual_size].copy_(sharded_weights)

        # pad the rest with zeros if needed
        # 对补齐的部分进行零填充
        if actual_size < shard_size:
            param_data[actual_size:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mask for tokens in this partition's range and within original vocab size
        # 计算掩码，确保只保留本地词表范围内的 token，并且不超过原始词表长度
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
                (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
                (x < self.num_embeddings)
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition) # 计算 token 在本地词表中的索引
        output = F.embedding(x, self.weight) # 词向量嵌入
        # 所谓的词向量嵌入实际上是拿 token id 即 x 去词表中查找相应的向量并返回
        # 虽然此处 x 因为 mask 将非本地词表部分置为 0，会去获取 0 号 token 的嵌入向量
        # 但是这些多余的词嵌入向量会在后面 all_reduce 时置为 0，因此不会影响最终结果

        if dist.get_world_size() > 1: # 分布式环境
            # need to mask again, otherwise the embedding for the out-of-range ids will be the embedding of id 0
            # 每个卡所本地保存的词表只能查出相应 mask 的词嵌入，要得到完整 output 需进行 all_reduce 合并
            # 自动广播，将不在本地词表范围内的 token 的嵌入置为 0，方便 all_reduce 求和
            output = mask.unsqueeze(1) * output
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

# LM head 即 Language Model head，通常是一个线性层，将 hidden dim 映射到词表大小的输出空间
# 可以视作 VocabParallelEmbedding 的反向版本，将 hidden dim 映射回词表大小的 logits
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
    # weight tying with embedding layer
    # 复用嵌入层的权重

    # x: [batch_size, seq_len, hidden_size]
    # x 是 layers 的输出，hidden_size 是模型的嵌入维度
    # weight: [vocab_size_per_partition, hidden_size]
    # weight 是本地词表的权重矩阵，vocab_size_per_partition 是每张卡维护的词表长度
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        """
        这里的 prefill 是指 prompt 填充阶段，通常在生成任务中，
        模型会先接收一个 prompt，对其进行全量线性变换为 hidden states，然后生成下一个 token。
        在 prefill 阶段，模型会处理整个 prompt，但在生成下一个 token时，
        只需要考虑 prompt 的最后一个 token，因为模型的输出是基于整个 prompt 的上下文信息来预测下一个 token 的概率分布。
        此处的加速逻辑是合并多个 prompt 的计算，因此只需要保留每个 prompt 的最后一个 token 的 hidden states 进行线性变换。
        """
        if context.is_prefill:
            # cu_seqlens_q = [0, 5, 8, 12] 左闭右开
            # cu_seqlens_q 实际上是每个 seq token 的起始位置索引
            # last_indices = [5, 8, 12] - 1 = [4, 7, 11]
            last_token = context.cu_seqlens_q[1:] - 1  # 排除起点位置
            x = x[last_token].contiguous()

        # logits: [batch_size, seq_len, vocab_size_per_partition]
        # logits 是经过线性层后的 output，本质上是词向量的相似性
        # F.linear 会自动转置 weight
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            # prepare for all_gather only for GPU 0 which is the main GPU
            # 只在 rank0 上缓存 all_logits，其他 GPU 上为 None
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # dist.gather 从所有 GPU 上收集 logits 存到 rank0 上的 all_logits 中
            dist.gather(logits, gather_list=all_logits, dst=0)
            # concatenate
            if self.tp_rank == 0:
                # [batch_size, seq_len, padded_vocab_size]
                # 拼接 logits
                logits = torch.cat(all_logits, dim=-1)
                # trim to original vocab size
                # 去除 padding 部分，保留原始词表大小
                logits = logits[..., :self.num_embeddings]

        return logits