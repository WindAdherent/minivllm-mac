import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *

# 由于是在单机中实现，此处不考虑分布式
def _mps_device(world_size: int, rank: int) -> torch.device:
    if world_size != 1:
        raise ValueError("macOS MPS ModelRunner requires world_size == 1")
    if rank != 0:
        raise ValueError("macOS MPS ModelRunner requires rank == 0")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable on this Mac")
    return torch.device("mps")


class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        self.device = _mps_device(self.world_size, self.rank)
        dist.init_process_group('gloo', "tcp://localhost:12345", world_size=self.world_size, rank=self.rank)

        # set model
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Move the model to its execution device before loading weights.
        self.model = self.model.to(self.device)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # 采样器根据 logits 和 temperature 采样 token
        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # 预热模型以了解推理过程的峰值额外内存
        self.warmup_model()
        # 根据预热结果将剩余内存分配给 KV cache
        self.allocate_kv_cache()

        torch.set_default_device(self.device)
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    # 工作进程读取命令
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        # 工作进程等待主进程即 rank 0 设置 event 写入共享内存
        self.event.wait()
        # 共享内存中前 4 个字节存储了数据长度，之后是 pickle 序列化的 (method_name, *args)
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 重置 event 等待下一次命令
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    # 主进程广播命令
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # 编码时会降维: (method_name, args) 调用时 args 是作为元组 -> (method_name, *args)
        # write_shm("run", (seqs, True)) -> ("run", seqs, True)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        # 唤醒所有等待共享内存的工作进程
        for event in self.event:
            event.set()

    # close shared memory and destroy the process group
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        torch.mps.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.mps.empty_cache()
        # 记录基准内存使用量：模型权重 + 模型持久缓冲区 + MPS 运行时分配
        baseline_memory = torch.mps.driver_allocated_memory()
        # 构造最大长度的模拟请求
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        # 触发最重的推理路径，MPS 会分配： Attention 临时张量 + MLP 临时张量 + logits + 算子工作区
        self.run(seqs, is_prefill=True)
        torch.mps.synchronize()
        warmup_memory = torch.mps.driver_allocated_memory()
        # 计算预热额外占用的内存量，分配 KV cache 的内存时会保留这部分，避免推理时 OOM
        self._warmup_memory_reserve = max(warmup_memory - baseline_memory, 0)
        torch.mps.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # 计算可用内存量，人为设定 GPU 占用上限，并考虑预热时的额外内存占用
        torch.mps.synchronize()
        memory_limit = int(torch.mps.recommended_max_memory() * self.config['gpu_memory_utilization'])
        driver_allocated_memory = torch.mps.driver_allocated_memory()
        warmup_memory_reserve = getattr(self, '_warmup_memory_reserve', 0)
        available_mem = memory_limit - driver_allocated_memory - warmup_memory_reserve
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads']
        if 'head_dim' in self.config:
            head_dim = self.config['head_dim']
        else:
            head_dim = self.config['hidden_size'] // self.config['num_heads']

        # 根据设定计算每个 KV cache block 的字节数
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        # 计算物理块数量，确保至少有一个块可用
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] max_cached_blocks: {self.config['max_cached_blocks']}")

        """
        直接为整个模型分配一个巨大的 KV cache 池
        维度 0：K 或 V
        维度 1：Transformer 层
        维度 2：物理 block_id
        维度 3：块内 token offset
        维度 4：KV Head
        维度 5：Head Dimension
        """
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        allocated_kv_cache = torch.zeros(2, num_layers, self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, dtype=self.default_dtype, device=self.device)
        layer_id = 0
        # 遍历模型的每一层，将对应的 KV cache block 分配给每个 Attention 模块
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        # 本次到底要计算哪些新 token
        input_ids = []
        # length: sum of all input_ids after prefix cache
        # 新 token 的 K/V 应写入哪个物理槽位
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        # 打包后的 Q 中，每个序列从哪里开始、在哪里结束
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        # 每个序列完整上下文的长度是多少
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        # 新 token 的 K/V 应写入哪个物理槽位
        block_tables = []
        # 逐序列处理，考虑 prefix cache 的影响
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            # 如果序列中有缓存的 token，则只取新 token 进行计算
            input_ids.extend(token_ids[num_cached_tokens:])
            # 构造 Query 和 Key 的长度信息，Query 只计算新 token，Key 计算完整上下文
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            # 构造 slot_mappings，指明新 token 的 K/V 应写入哪个物理槽位
            if seq.block_table:
                # 跳过已经缓存的完整块
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    # 非最后一个块一定是完整块，直接映射整个块
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    # 最后一个块可能是非完整块，只映射实际的 token 数量    
                    else:
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        # 当 Q 的长度小于 K 的长度时，表明已经有前缀缓存，需要构造 block_tables，指明新 token 的 K/V 应从哪些物理槽位读取
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            # 对每个序列的 block_table 进行填充，使其长度一致，无效位置用 -1 填充
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, device=self.device)

        # 将 prefill 的上下文信息传递给模型，供每个 Attention 模块使用
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, device=self.device),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device) if block_tables else None,
        )
        return input_ids


    # prepare input data for decoding
    # 准备逐 token 解码的输入数据
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            # 逐序列处理，取出最后一个 token 作为解码输入
            input_ids.append(seq.last_token)
            # context_lens 记录每个序列的完整上下文长度，供 Attention 模块使用
            context_lens.append(len(seq))
            # slot_mappings 记录每个序列的最后一个 token 应写入哪个物理槽位
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            # Decode 必须始终提供 block_tables，即使没有前缀缓存，也要传递填充为空的 block_tables 以保持接口一致
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, device=self.device),
            context_lens=torch.tensor(context_lens, dtype=torch.long, device=self.device),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device) if block_tables else None,
        )
        return input_ids    

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, device=self.device)

    # 关闭梯度记录、autograd 计算图、部分 Tensor version tracking，相较于 torch.no_grad() 更节省内存和计算开销
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        return self.model.compute_logits(hidden_states)


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids
