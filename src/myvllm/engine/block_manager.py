import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

# 每个 Block 是一个物理 KV Cache 块的 CPU 侧元数据
class Block:
    def __init__(self, block_id):
        # 物理块编号，实际的 slot mapping 中 token_slot = block_id * block_size + offset_in_block
        self.block_id = block_id
        # hash 值，用于 prefix caching
        self.hash = -1 
        # 引用计数，表示有多少个 sequence 正在使用这个 block
        self.ref_count = 0
        # 记录 KV Cache 中的 token ids，主要用于 prefix caching 正确性校验
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # 块大小，即每个 block 中的 token 数量
        self.block_size: int = block_size
        # 建立所有物理块的元数据
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 前缀缓存索引
        self.hash_to_block_id: dict[int, int] = {}
        # 空闲块队列，初始时包含所有块的编号
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 当前正在使用的物理块，初始为空
        self.used_block_ids: set[int] = set()

    # 输入 token_ids 和 prefix_hash_value，计算 hash 值
    # 代表从序列开头到当前块结束的完整前缀，而不只是当前块的 token
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # 把物理块标记为已使用
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    # 把物理块标记为未使用
    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 检查是否有足够的空闲块来分配给序列
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # 为一个新序列分配全部块
    def allocate(self, seq: Sequence) -> None:
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # 只计算完整块的 hash 值，部分块的 hash 值为 -1，表示不缓存；部分块还会追加 token，每次追加都会导致 hash 值变化，因此不适合缓存
            # 只有完整块能够共享；最后一个部分块必须由每个序列独占，因此不需要 Copy-on-Write
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # if cache miss or hash collision
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # update sequence information
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                else:
                    # update block information
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # cache miss
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)
        
    # 释放一个序列占用的所有块
    def deallocate(self, seq: Sequence) -> None:
        # 将 block 映射表中的所有 block 的引用计数减 1，如果引用计数为 0，则释放该 block
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 清空序列的 block_table 和 num_cached_tokens 映射
        seq.block_table = []
        seq.num_cached_tokens = 0

    # 在下一轮 Decode 处理最新 token 时，是否会需要一个新物理块
    def can_append(self, seq: Sequence) -> bool:
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    # Decode 过程中维护最后一个块
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # 情况 1：最后一个 block 刚刚变满，计算链式 hash 值并更新 block 的 token_ids
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # 情况 2：新 token 是新逻辑块的第一个 token，分配一个新的物理块
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # 情况 3：继续填充当前部分块，更新 token_ids，但不更新 hash 值，因为部分块的 hash 值为 -1
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"