from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy

# 表示一条请求的生命周期
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.block_size = block_size # block 分块大小
        # record sequence id
        self.seq_id = next(Sequence.counter)
        # status
        self.status = SequenceStatus.WAITING

        # token ids, need copy so that it is a new list, won't be affected by outside changes
        # 所有 token
        self.token_ids = copy(token_ids)
        # 最后一个 token
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # 总 token 数量
        self.num_tokens = len(self.token_ids)
        # 原始 prompt 的 token 数量
        self.num_prompt_tokens = len(self.token_ids)
        # 以上四个 token 相关属性用于保存 token 进度

        # num_cached_tokens = 0
        # 已缓存的 token 数量，主要用于在 prefill 阶段和 decode 阶段之间传递信息
        self.num_cached_tokens = 0
        # 记录已使用的 block id
        self.block_table = []

        # sampling_params' related things
        self.temperature = sampling_params.temperature # 采样温度
        self.max_tokens = sampling_params.max_tokens # 允许生成的最大 token 数量
        self.ignore_eos = sampling_params.ignore_eos # 是否忽略 EOS token
        self.max_model_length = sampling_params.max_model_length # 允许的总 token 数量上限，即模型的最大上下文长度

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        full_blocks = int(math.floor(self.num_tokens / self.block_size))
        return len(self.token_ids[full_blocks * self.block_size : ])

    # 返回第 i 个 block 的 token ids
    def block(self, i):
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    # 序列化函数
    def __getstate__(self):
        return (
            self.num_tokens, 
            self.num_prompt_tokens, 
            self.num_cached_tokens, 
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    # 反序列化函数
    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            last_token_or_ids
        ) = state
        # Check if this is prefill (num_completion_tokens == 0) or decode phase
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        if num_completion_tokens == 0:
            # Prefill: last_token_or_ids is the full token_ids list
            self.token_ids = last_token_or_ids
        else:
            # Decode: last_token_or_ids is just the last token
            self.token_ids = [last_token_or_ids]
        # Restore last_token attribute
        self.last_token = self.token_ids[-1] if self.token_ids else None