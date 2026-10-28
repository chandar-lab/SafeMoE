"""CPU tests of the SafeMoE layer on a tiny random Mistral (python -m pytest tests)."""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM

from safemoe.moe import compose_safemoe, load_safemoe, moe_layers, set_routing_mode, set_trainable


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    config = MistralConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    MistralForCausalLM(config).save_pretrained(root / "base")
    adapters = []
    for i in range(3):
        torch.manual_seed(i + 1)
        lora = LoraConfig(r=4, lora_alpha=8, target_modules=["gate_proj", "up_proj", "down_proj"],
                          init_lora_weights=False, task_type="CAUSAL_LM")
        peft_model = get_peft_model(MistralForCausalLM.from_pretrained(root / "base"), lora)
        peft_model.save_pretrained(root / f"expert{i}")
        adapters.append(str(root / f"expert{i}"))
    return str(root / "base"), adapters, root


def _reference_output(base, adapter, ids):
    """Dense output of the base model with one LoRA expert applied by PEFT."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(MistralForCausalLM.from_pretrained(base), adapter).eval()
    return model(ids).logits


def test_single_expert_matches_peft(tiny):
    base, adapters, _ = tiny
    model = compose_safemoe(base, adapters[:1], num_experts_per_tok=1, torch_dtype=torch.float32).eval()
    ids = torch.randint(0, 128, (2, 7))
    # With one expert and top-1, the softmax weight is 1, so the MoE equals the LoRA model.
    torch.testing.assert_close(model(ids).logits, _reference_output(base, adapters[0], ids))


def test_save_load_roundtrip_and_trainable(tiny):
    base, adapters, root = tiny
    model = compose_safemoe(base, adapters, num_experts_per_tok=2, torch_dtype=torch.float32).eval()
    assert len(moe_layers(model)) == 2 * 3
    routers = sum(layer.gate.weight.numel() for _, layer in moe_layers(model))
    assert set_trainable(model, "router") == routers
    gate_proj = sum(p.numel() for n, p in model.named_parameters() if "gate_proj" in n and not n.endswith(".gate.weight"))
    assert set_trainable(model) == routers + gate_proj  # rule of the original training script
    ids = torch.randint(0, 128, (2, 7))
    expected = model(ids).logits
    model.save_pretrained(root / "moe")
    loaded = load_safemoe(str(root / "moe"), torch_dtype=torch.float32).eval()
    torch.testing.assert_close(loaded(ids).logits, expected)


def test_top1_routing_has_router_gradients(tiny):
    base, adapters, _ = tiny
    model = compose_safemoe(base, adapters, num_experts_per_tok=1, torch_dtype=torch.float32)
    set_trainable(model, "router")
    ids = torch.randint(0, 128, (2, 7))

    model(ids).logits.square().mean().backward()

    assert any(
        layer.gate.weight.grad is not None and layer.gate.weight.grad.abs().sum() > 0
        for _, layer in moe_layers(model)
    )


@pytest.mark.parametrize("mode", ["first_layer", "last_token"])
def test_routing_modes_generate(tiny, mode):
    base, adapters, _ = tiny
    model = compose_safemoe(base, adapters, num_experts_per_tok=1, torch_dtype=torch.float32).eval()
    set_routing_mode(model, mode)
    out = model.generate(torch.randint(0, 128, (1, 5)), max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert out.shape == (1, 9)
    if mode == "first_layer":
        routers = [layer for _, layer in moe_layers(model) if layer.is_router]
        assert len(routers) == 3  # gate_proj, up_proj, down_proj of block 0


def test_last_token_routing_uses_each_prompt_boundary(tiny):
    base, adapters, _ = tiny
    model = compose_safemoe(base, adapters, num_experts_per_tok=1, torch_dtype=torch.float32).train()
    set_routing_mode(model, "last_token")
    layer = moe_layers(model)[0][1]
    layer_input = {}
    route_input = {}

    def capture_layer_input(_module, args):
        layer_input["value"] = args[0].detach().clone()

    original_route = layer.route

    def capture_route(x):
        route_input["value"] = x.detach().clone()
        return original_route(x)

    layer.register_forward_pre_hook(capture_layer_input)
    layer.route = capture_route
    ids = torch.randint(0, 128, (2, 6))
    prompt_lengths = torch.tensor([2, 4])
    model(input_ids=ids, attention_mask=torch.ones_like(ids), prompt_lengths=prompt_lengths)

    expected = layer_input["value"][torch.arange(2), prompt_lengths - 1].unsqueeze(1)
    torch.testing.assert_close(route_input["value"], expected)
