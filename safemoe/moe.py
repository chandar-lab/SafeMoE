"""Sparse mixture of LoRA experts (Section 3.3).

Every target linear layer W of the base model (by default the MLP projections
gate_proj / up_proj / down_proj of every decoder block) is replaced by

    h = W x + sum_{i in top-k(r)} w_i * B_i A_i x * scaling_i,   r = V^T x,
    w = softmax(r[top-k])

where (A_i, B_i) are frozen LoRA experts and V is the only trainable router.

The layer is derived from mergoo's LoRAMoeLayer (LGPL-3.0,
https://github.com/Leeroo-AI/mergoo) and keeps its parameter names
(`base_layer`, `lora_A.<i>`, `lora_B.<i>`, `gate`), so checkpoints written by
the original mergoo-based code load unchanged with `load_safemoe`.

Instead of vendoring a patched copy of each HF modeling file, the MoE layers
are injected into a stock transformers model, which works for every decoder
that exposes `model.model.layers` (Mistral, Qwen2, Llama, ...).
"""

import json
import math
import os

import torch
import torch.nn.functional as F
from peft.utils import load_peft_weights
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

ROUTING_MODES = ("per_token", "first_layer", "last_token")


class LoRAMoeLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias,
        adapter_configs,
        num_experts_per_tok=2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = len(adapter_configs)
        self.num_experts_per_tok = num_experts_per_tok

        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.gate = nn.Linear(in_features, self.num_experts, bias=False)
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.scaling = {}
        for ix, cfg in enumerate(adapter_configs):
            name = str(ix)
            r, alpha = cfg["r"], cfg["lora_alpha"]
            self.lora_A[name] = nn.Linear(in_features, r, bias=False)
            self.lora_B[name] = nn.Linear(r, out_features, bias=False)
            dropout = cfg.get("lora_dropout", 0.0)
            self.lora_dropout[name] = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.scaling[name] = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)

        # Routing variants of Section 4.2.8; see `set_routing_mode`.
        self.routing_mode = "per_token"
        self.routing_key = None
        self.routing_context = None
        self.routing_prompt_lengths = None
        self.is_router = True
        self._cached_routing = None

    def route(self, x):
        """Top-k experts of r = V^T x and their softmax-normalised weights, each (batch, seq, k)."""
        logits = self.gate(x)
        selected_logits, selected = torch.topk(logits, self.num_experts_per_tok)
        if self.num_experts_per_tok == 1:
            selected_probs = F.softmax(logits, dim=-1, dtype=torch.float).gather(-1, selected).to(x.dtype)
            weights = torch.ones_like(selected_probs) + (selected_probs - selected_probs.detach())
        else:
            weights = F.softmax(selected_logits, dim=-1, dtype=torch.float).to(x.dtype)
        return weights, selected

    def _routing_for(self, x):
        if self.routing_mode == "per_token":
            return self.route(x)

        if self.routing_mode == "first_layer":
            # Block 0 routes every token; deeper blocks reuse its decision.
            if self.is_router:
                weights, selected = self.route(x)
                self.routing_context[self.routing_key] = (weights, selected)
                return weights, selected
            if self.routing_key not in self.routing_context:
                raise RuntimeError("first_layer routing: block 0 has not run for this forward pass")
            weights, selected = self.routing_context[self.routing_key]
            return weights.to(x.dtype), selected

        # last_token: route once from the last prompt token and broadcast the
        # decision to every position. During generation (seq_len == 1) the
        # decision made at prefill is reused.
        if self.training or x.shape[1] > 1 or self._cached_routing is None:
            route_input = x[:, -1:, :]
            if self.routing_prompt_lengths is not None:
                indices = self.routing_prompt_lengths.to(device=x.device, dtype=torch.long)
                indices = (indices - 1).clamp(min=0, max=x.shape[1] - 1)
                route_input = x[torch.arange(x.shape[0], device=x.device), indices].unsqueeze(1)
            weights, selected = self.route(route_input)
            if not self.training:
                self._cached_routing = (weights, selected)
        else:
            weights, selected = self._cached_routing
        seq = x.shape[1]
        return weights.expand(-1, seq, -1), selected.expand(-1, seq, -1)

    def forward(self, x, *args, **kwargs):
        previous_dtype = x.dtype
        weights, selected = self._routing_for(x)
        result = self.base_layer(x, *args, **kwargs)
        for ix in range(self.num_experts):
            name = str(ix)
            batch_idx, tok_idx, slot_idx = torch.where(selected == ix)
            if batch_idx.numel() == 0:
                continue
            lora_A = self.lora_A[name]
            x_e = x[batch_idx, tok_idx].to(lora_A.weight.dtype)
            delta = self.lora_B[name](lora_A(self.lora_dropout[name](x_e))) * self.scaling[name]
            result[batch_idx, tok_idx] += delta * weights[batch_idx, tok_idx, slot_idx].unsqueeze(-1)
        return result.to(previous_dtype)

    # Some HF code paths read `.weight` / `.bias` of projection layers.
    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


def _decoder_layers(model):
    return model.model.layers


def _moe_config_fields(model_config):
    return {
        "num_experts": model_config.num_experts,
        "num_experts_per_tok": model_config.num_experts_per_tok,
        "router_layers": list(model_config.router_layers),
        "router_layers_index": list(model_config.router_layers_index),
        "adapter_configs": model_config.adapter_configs,
    }


def inject_moe_layers(model, copy_base_weights=True):
    """Replace target linears with LoRAMoeLayer, using the MoE fields of model.config."""
    fields = _moe_config_fields(model.config)
    layers_index = set(fields["router_layers_index"])
    replaced = 0
    for layer_idx, block in enumerate(_decoder_layers(model)):
        if layer_idx not in layers_index:
            continue
        for name, module in list(block.named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in fields["router_layers"] or not isinstance(module, nn.Linear):
                continue
            moe = LoRAMoeLayer(
                module.in_features,
                module.out_features,
                module.bias is not None,
                fields["adapter_configs"],
                fields["num_experts_per_tok"],
            ).to(device=module.weight.device, dtype=module.weight.dtype)
            if copy_base_weights:
                moe.base_layer = module
            parent = block.get_submodule(name.rsplit(".", 1)[0]) if "." in name else block
            setattr(parent, leaf, moe)
            replaced += 1
    return replaced


def moe_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRAMoeLayer)]


def _adapter_config_dict(adapter_path):
    # Read the JSON directly: PeftConfig.from_pretrained drops fields written by
    # other PEFT versions (e.g. target_modules of adapters saved by PEFT >= 0.15).
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        cfg = json.load(f)
    if not cfg.get("target_modules"):
        raise ValueError(f"{adapter_path}: adapter_config.json has no target_modules")
    return cfg


def compose_safemoe(
    base_model,
    adapter_paths,
    num_experts_per_tok=2,
    torch_dtype=torch.bfloat16,
    seed=42,
):
    """Build a SafeMoE model in memory from a base model and a list of LoRA experts (top-k softmax routing).

    The router weights are initialised N(0, initializer_range), which is what
    transformers' `_init_weights` gave the gate in the original pipeline.
    """
    adapter_configs = [_adapter_config_dict(p) for p in adapter_paths]
    target_modules = sorted(adapter_configs[0]["target_modules"])
    for path, cfg in zip(adapter_paths, adapter_configs):
        if sorted(cfg["target_modules"]) != target_modules:
            raise ValueError(f"{path} targets {cfg['target_modules']}, expected {target_modules}")

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)
    model.config.update(
        {
            "num_experts": len(adapter_paths),
            "num_experts_per_tok": num_experts_per_tok,
            "router_layers": target_modules,
            "router_layers_index": list(range(model.config.num_hidden_layers)),
            "adapter_configs": adapter_configs,
            "safemoe_base_model": base_model,
        }
    )
    inject_moe_layers(model)

    modules = dict(model.named_modules())
    for ix, path in enumerate(adapter_paths):
        loaded = 0
        for key, tensor in load_peft_weights(path, device="cpu").items():
            # base_model.model.model.layers.3.mlp.up_proj.lora_A.weight
            #   -> model.layers.3.mlp.up_proj  /  lora_A
            module_path, lora_name, _ = key.removeprefix("base_model.model.").rsplit(".", 2)
            if module_path not in modules:
                continue
            target = getattr(modules[module_path], lora_name)[str(ix)]
            target.weight.data.copy_(tensor)
            loaded += 1
        if loaded == 0:
            raise ValueError(f"No LoRA weights of {path} matched the model")

    generator = torch.Generator().manual_seed(seed)
    for _, layer in moe_layers(model):
        w = torch.empty(layer.gate.weight.shape, dtype=torch.float32)
        w.normal_(0.0, model.config.initializer_range, generator=generator)
        layer.gate.weight.data.copy_(w)
    return model


_SAFEMOE_CLASSES = {}


def _safemoe_class(config):
    base_cls = MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]
    if base_cls not in _SAFEMOE_CLASSES:

        def __init__(self, config):
            base_cls.__init__(self, config)
            inject_moe_layers(self)

        _SAFEMOE_CLASSES[base_cls] = type(f"SafeMoE{base_cls.__name__}", (base_cls,), {"__init__": __init__})
    return _SAFEMOE_CLASSES[base_cls]


def is_safemoe_checkpoint(path):
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as f:
        return "adapter_configs" in json.load(f)


def load_safemoe(path, torch_dtype=torch.bfloat16, **kwargs):
    """Load a SafeMoE checkpoint (also loads checkpoints from the original mergoo code)."""
    config = AutoConfig.from_pretrained(path)
    return _safemoe_class(config).from_pretrained(path, config=config, torch_dtype=torch_dtype, **kwargs)


def set_trainable(model, mode="gate_substring"):
    """Select the trainable parameters for router training. Returns their number.

    mode="gate_substring"  every parameter whose name contains "gate", the rule of
                           the original router-training script: the routers V
                           (`*.gate.weight`) and, because the MLP sublayer is named
                           gate_proj, also the base gate_proj weights and the
                           experts' gate_proj LoRAs.
    mode="router"          only the routers (`*.gate.weight`); used by the
                           random-expert and routing-variant ablations.
    """
    if mode not in ("router", "gate_substring"):
        raise ValueError(f"unknown trainable mode {mode!r}")
    n = 0
    for name, param in model.named_parameters():
        trainable = name.endswith(".gate.weight") if mode == "router" else "gate" in name
        param.requires_grad_(trainable)
        n += param.numel() if trainable else 0
    return n


def set_routing_mode(model, mode):
    """Switch every MoE layer to one of ROUTING_MODES (Section 4.2.8).

    first_layer: only block 0 routes; deeper blocks reuse its decision for the
                 same target module, so only block-0 routers should be trained.
    last_token:  each block routes once from the last token and broadcasts.
    """
    if mode not in ROUTING_MODES:
        raise ValueError(f"routing mode must be one of {ROUTING_MODES}")
    context = {}
    first_block = min(model.config.router_layers_index)
    layers = moe_layers(model)
    for name, layer in layers:
        layer.routing_mode = mode
        layer.routing_key = name.rsplit(".", 1)[-1]
        layer.routing_context = context
        layer.is_router = f".layers.{first_block}." in f".{name}."
        layer._cached_routing = None

    def reset(*_args, **_kwargs):
        context.clear()

    if mode == "first_layer" and not getattr(model, "_safemoe_reset_hook", None):
        model._safemoe_reset_hook = model.register_forward_pre_hook(reset)

    if not getattr(model, "_safemoe_prompt_hook", None):
        def set_prompt_lengths(_module, _args, kwargs):
            prompt_lengths = kwargs.pop("prompt_lengths", None)
            for _, layer in layers:
                layer.routing_prompt_lengths = prompt_lengths
            return _args, kwargs

        model._safemoe_prompt_hook = model.register_forward_pre_hook(set_prompt_lengths, with_kwargs=True)
    return model


def reset_routing_cache(model):
    """Clear the last-token routing cache; call before each new generate()."""
    for _, layer in moe_layers(model):
        layer._cached_routing = None
