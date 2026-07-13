import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


MODEL_RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "myvllm" / "engine" / "model_runner.py"
)


class DummyModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.moved_to = None

    def to(self, device):
        self.moved_to = torch.device(device)
        return self

    def forward(self, input_ids):
        return input_ids

    def compute_logits(self, hidden_states):
        return hidden_states


class DummySampler(torch.nn.Module):
    def forward(self, logits, temperature=None):
        return logits.argmax(dim=-1).tolist()


class DummySequence:
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size


def load_model_runner(monkeypatch):
    qwen3_module = ModuleType("myvllm.models.qwen3")
    qwen3_module.Qwen3ForCausalLM = DummyModel
    monkeypatch.setitem(sys.modules, qwen3_module.__name__, qwen3_module)

    llama_module = ModuleType("myvllm.models.llama")
    llama_module.LlamaForCausalLM = DummyModel
    monkeypatch.setitem(sys.modules, llama_module.__name__, llama_module)

    sampler_module = ModuleType("myvllm.layers.sampler")
    sampler_module.SamplerLayer = DummySampler
    monkeypatch.setitem(sys.modules, sampler_module.__name__, sampler_module)

    sequence_module = ModuleType("myvllm.engine.sequence")
    sequence_module.Sequence = DummySequence
    monkeypatch.setitem(sys.modules, sequence_module.__name__, sequence_module)

    context_state = SimpleNamespace(context=None)

    def set_context(**kwargs):
        context_state.context = SimpleNamespace(**kwargs)

    def get_context():
        return context_state.context

    def reset_context():
        context_state.context = None

    utils_module = ModuleType("myvllm.utils")
    utils_module.__path__ = []
    utils_module.__all__ = ["set_context", "get_context", "reset_context"]
    utils_module.set_context = set_context
    utils_module.get_context = get_context
    utils_module.reset_context = reset_context
    monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)

    loader_module = ModuleType("myvllm.utils.loader")
    loader_module.load_weights_from_checkpoint = lambda model, path: None
    monkeypatch.setitem(sys.modules, loader_module.__name__, loader_module)

    spec = importlib.util.spec_from_file_location(
        "isolated_model_runner", MODEL_RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._test_context_state = context_state
    return module


def qwen_config():
    return {
        "block_size": 2,
        "world_size": 1,
        "enforce_eager": True,
        "model_name_or_path": "/tmp/Qwen3-0.6B",
        "vocab_size": 32,
        "hidden_size": 8,
        "num_heads": 2,
        "head_dim": 4,
        "scale": 0.5,
        "num_kv_heads": 2,
        "rms_norm_epsilon": 1e-5,
        "qkv_bias": False,
        "base": 10000,
        "max_position": 32,
        "intermediate_size": 16,
        "ffn_bias": False,
        "num_layers": 1,
        "tie_word_embeddings": False,
    }


def test_mps_device_rejects_unsupported_runtime(monkeypatch):
    module = load_model_runner(monkeypatch)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    with pytest.raises(ValueError, match="world_size == 1"):
        module._mps_device(2, 0)
    with pytest.raises(ValueError, match="rank == 0"):
        module._mps_device(1, 1)

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        module._mps_device(1, 0)


def test_constructor_uses_gloo_and_moves_model_to_mps(monkeypatch):
    module = load_model_runner(monkeypatch)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    init_calls = []
    default_devices = []
    monkeypatch.setattr(
        module.dist,
        "init_process_group",
        lambda backend, init_method, world_size, rank: init_calls.append(
            (backend, init_method, world_size, rank)
        ),
    )
    monkeypatch.setattr(
        module.torch,
        "set_default_device",
        lambda device: default_devices.append(device),
    )
    monkeypatch.setattr(module.ModelRunner, "warmup_model", lambda self: None)
    monkeypatch.setattr(module.ModelRunner, "allocate_kv_cache", lambda self: None)

    runner = module.ModelRunner(qwen_config(), rank=0, event=None)

    assert runner.device == torch.device("mps")
    assert runner.model.moved_to == torch.device("mps")
    assert init_calls == [("gloo", "tcp://localhost:12345", 1, 0)]
    assert default_devices == [torch.device("mps")]
