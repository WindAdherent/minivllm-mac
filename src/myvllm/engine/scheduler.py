from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)

    # 返回 scheduled_sequences, is_prefill，其中 is_prefill 表示 prefill 阶段还是 decode 阶段
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # try schedule for prefilling from waiting queue if not exceeding limits
        # 等待队列非空，且当前 batch 中的序列数未超过上限
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            # 查看等待队列的第一个序列，检查剩余资源是否足够调度
            seq = self.waiting[0]
            # 检查 KV cache 和 prefill token bugdet 是否足够
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
        # prefill 拥有更高的优先级，如果有序列被调度了，则直接返回
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        # 调度 Decode 阶段
        while self.running:
            # 从 running 队列中取出队首序列，尝试调度
            seq = self.running.popleft()
            # 检查是否有空间处理最新 token
            # 如果 KV cache 空间不足，则将该序列放回队列头部，并尝试抢占
            if not self.block_manager.can_append(seq):
                # 如果当前队列中有其他序列正在运行，则将该序列放回队列头部，并尝试抢占最后一个序列的资源
                if self.running:
                    self.running.appendleft(seq)
                    # 将队尾的序列弹出，并释放资源
                    self.preempt(self.running.pop())
                    # 双端队列 pop()是从右边弹出，appendleft()是从左边插入
                # 如果当前队列中没有其他序列正在运行，则抢占自身，将自身放入 waiting 队首，释放自身所有 KV cache
                else:
                    self.preempt(seq)
                    break
            # 如果空间充足，尝试调度
            else:
                # 检查是否超过了当前 batch 的 token 数量限制，或者超过了当前 batch 的序列数量限制
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    # 如果超过了限制，则将该序列放回 running 队列头部，并结束调度
                    self.running.appendleft(seq)
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False

    # 抢占一个 sequence 的资源，将其放回 waiting 队列头部，并释放其 KV cache
    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)