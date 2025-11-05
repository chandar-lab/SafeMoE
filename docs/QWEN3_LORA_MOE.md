# Qwen3-8B LoRA MoE Implementation

## Overview

This document explains the implementation of **Mixture of LoRA Experts (MoE)** for **Qwen3-8B** in the mergoo library. This allows you to combine multiple LoRA adapters trained on different tasks into a single model that can dynamically route tokens to the most appropriate expert.

## What is LoRA MoE?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method that adds small trainable matrices to frozen pretrained models.

**Mixture of Experts (MoE)** is an architecture where multiple "expert" networks specialize in different aspects of the task, and a routing mechanism decides which experts to use for each input.

**LoRA MoE** combines these concepts:
- Multiple LoRA adapters (experts) are trained on different tasks/domains
- A learned router dynamically selects which experts to activate for each token
- This allows a single model to leverage specialized knowledge from multiple fine-tuned adapters

## Architecture

### Key Components

1. **Base Model**: Qwen3-8B pretrained model (frozen weights)
2. **LoRA Experts**: Multiple LoRA adapters, each fine-tuned on a specific task
3. **Router Layers**: Learned gating networks that select which experts to use
4. **MoE Layers**: Replace standard linear layers with mixture of expert layers

### Modified Layers

The implementation modifies the following Qwen3 components to support MoE:

#### MLP Layers (Feed-Forward Network)
- `gate_proj`: Projects hidden states to intermediate dimension with gating
- `up_proj`: Projects hidden states to intermediate dimension  
- `down_proj`: Projects back to hidden dimension

#### Attention Layers
- `q_proj`: Query projection
- `k_proj`: Key projection
- `v_proj`: Value projection
- `o_proj`: Output projection

### How It Works

For each layer specified in the configuration:

1. **Standard Layer** (if not in MoE config):
   ```
   output = Linear(input)
   ```

2. **MoE Layer** (if in MoE config):
   ```
   # Router selects top-k experts
   router_logits = Router(input)
   expert_weights, expert_indices = topk(router_logits, k=num_experts_per_tok)
   
   # Base model output
   base_output = BaseLinear(input)
   
   # Add weighted expert outputs
   for expert_idx in expert_indices:
       lora_output = LoRA_B[expert_idx](LoRA_A[expert_idx](input))
       base_output += expert_weights[expert_idx] * lora_output
   
   output = base_output
   ```

## Implementation Details

### File Structure

```
mergoo/
├── models/
│   ├── modeling_qwen3.py          # Main Qwen3 model with MoE support
│   └── __init__.py                # Exports Qwen3 classes
├── compose_layers.py              # MoE layer implementations
└── composers/
    └── composer_lora_moe.py       # LoRA MoE composition logic

examples/
└── compose_lora_qwen3.py          # Example usage script

docs/
└── QWEN3_LORA_MOE.md             # This file
```

### Key Classes

#### `Qwen3MLP` (in `modeling_qwen3.py`)
```python
class Qwen3MLP(nn.Module):
    def __init__(self, config, layer_idx=None):
        # Uses convert_linear_to_moe to conditionally create MoE layers
        self.gate_proj = convert_linear_to_moe("gate_proj", config, layer_idx, ...)
        self.up_proj = convert_linear_to_moe("up_proj", config, layer_idx, ...)
        self.down_proj = convert_linear_to_moe("down_proj", config, layer_idx, ...)
```

#### `Qwen3Attention` (in `modeling_qwen3.py`)
```python
class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        # Attention projections can also be MoE
        self.q_proj = convert_linear_to_moe("q_proj", config, layer_idx, ...)
        self.k_proj = convert_linear_to_moe("k_proj", config, layer_idx, ...)
        self.v_proj = convert_linear_to_moe("v_proj", config, layer_idx, ...)
        self.o_proj = convert_linear_to_moe("o_proj", config, layer_idx, ...)
        
        # Qwen3-specific: Q/K normalization
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

#### `convert_linear_to_moe` (in `compose_layers.py`)
```python
def convert_linear_to_moe(name, config, layer_idx, in_features, out_features, bias):
    # Check if this layer should be MoE
    if (layer_idx in config.router_layers_index) and (name in config.router_layers):
        if hasattr(config, "adapter_configs"):
            # Create LoRA MoE layer
            return LoRAMoeLayer(config, in_features, out_features, bias)
        else:
            # Create standard MoE layer
            return MoeLayer(in_features, out_features, bias, ...)
    # Otherwise return standard linear layer
    return nn.Linear(in_features, out_features, bias=bias)
```

### Qwen3-Specific Features

The implementation includes Qwen3-specific architectural features:

1. **Q/K Normalization**: Unlike Qwen2, Qwen3 applies RMSNorm to query and key states
   ```python
   query_states = self.q_norm(query_states)
   key_states = self.k_norm(key_states)
   ```

2. **Sliding Window Attention**: Support for sliding window attention in certain layers
   ```python
   self.sliding_window = config.sliding_window if layer_types[layer_idx] == "sliding_attention" else None
   ```

3. **No Attention Bias**: Qwen3 uses `attention_bias=False` by default (unlike Qwen2)

4. **Layer Types**: Qwen3 config includes `layer_types` to specify attention patterns per layer

## Usage

### 1. Prepare LoRA Adapters

First, fine-tune multiple LoRA adapters on different tasks:

```python
# Example: Fine-tune on math problems
python lora_finetune.py \
    --base_model Qwen/Qwen3-8B \
    --dataset math_dataset.json \
    --output_dir loras/qwen3-math

# Example: Fine-tune on code
python lora_finetune.py \
    --base_model Qwen/Qwen3-8B \
    --dataset code_dataset.json \
    --output_dir loras/qwen3-code
```

### 2. Compose LoRA MoE Model

Create a configuration and compose the experts:

```python
import torch
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM

config = {
    "model_type": "qwen3",
    "num_experts_per_tok": 2,
    "base_model": "Qwen/Qwen3-8B",
    "experts": [
        {"expert_name": "math", "model_id": "loras/qwen3-math"},
        {"expert_name": "code", "model_id": "loras/qwen3-code"},
        {"expert_name": "reasoning", "model_id": "loras/qwen3-reasoning"},
    ],
    "router_layers": ["gate_proj", "up_proj", "down_proj"],
    "router_layers_index": list(range(28)),  # All 28 layers
}

composer = ComposeExperts(config, torch_dtype=torch.bfloat16, model_cls=Qwen3ForCausalLM)
composer.compose()
composer.save_checkpoint("qwen3_lora_moe")
```

### 3. Use the Composed Model

Load and use the model for inference:

```python
from transformers import AutoTokenizer
from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM

# Load model and tokenizer
model = Qwen3ForCausalLM.from_pretrained("qwen3_lora_moe")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Generate text
prompt = "Solve this math problem: What is 15% of 240?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Configuration Options

### `model_type`
- **Type**: `str`
- **Value**: `"qwen3"`
- **Description**: Specifies the model architecture

### `num_experts_per_tok`
- **Type**: `int`
- **Default**: `2`
- **Description**: Number of experts to activate per token. Higher values use more experts but increase computation.

### `base_model`
- **Type**: `str`
- **Example**: `"Qwen/Qwen3-8B"`
- **Description**: HuggingFace model ID or local path to the base Qwen3 model

### `experts`
- **Type**: `List[Dict]`
- **Description**: List of LoRA adapters to use as experts
- **Format**: Each dict should have:
  - `expert_name`: Identifier for the expert
  - `model_id`: Path to the LoRA adapter

### `router_layers`
- **Type**: `List[str]`
- **Options**: `["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]`
- **Description**: Which layer types to convert to MoE
- **Common Choices**:
  - MLP only: `["gate_proj", "up_proj", "down_proj"]`
  - Attention only: `["q_proj", "k_proj", "v_proj", "o_proj"]`
  - Both: All of the above

### `router_layers_index`
- **Type**: `List[int]`
- **Range**: `0` to `27` (for Qwen3-8B with 28 layers)
- **Description**: Which transformer layers to apply MoE
- **Examples**:
  - All layers: `list(range(28))`
  - Last 10 layers: `list(range(18, 28))`
  - Every other layer: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]`

## Differences from Qwen2 Implementation

| Feature | Qwen2 | Qwen3 |
|---------|-------|-------|
| Attention Bias | `bias=True` | `bias=False` |
| Q/K Normalization | No | Yes (RMSNorm on head_dim) |
| Sliding Window | Basic support | Layer-specific via `layer_types` |
| Config Parameter | `head_dim` optional | `head_dim` explicit |
| Layer Types | Not used | `layer_types` list per layer |

## Best Practices

1. **Expert Selection**: Train LoRA adapters on diverse, complementary tasks
2. **Layer Selection**: Start with MLP layers only, add attention if needed
3. **Number of Experts**: 3-8 experts is usually a good range
4. **Experts per Token**: Start with 2, increase if model needs more capacity
5. **Layer Indices**: Apply MoE to later layers for better task-specific routing

## Troubleshooting

### Issue: Out of Memory
- **Solution**: Reduce `num_experts_per_tok` or number of experts
- **Solution**: Apply MoE to fewer layers (reduce `router_layers_index`)
- **Solution**: Use `torch_dtype=torch.float16` or `torch.bfloat16`

### Issue: Poor Performance
- **Solution**: Ensure LoRA adapters are properly trained
- **Solution**: Try different `router_layers` combinations
- **Solution**: Increase `num_experts_per_tok` for more expert utilization

### Issue: Model Not Loading
- **Solution**: Verify all LoRA adapter paths are correct
- **Solution**: Ensure base model is Qwen3-8B (not Qwen2 or other variants)
- **Solution**: Check that `model_cls=Qwen3ForCausalLM` is specified

## References

- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-8B)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Mixture of Experts](https://arxiv.org/abs/1701.06538)
- [Mergoo Library](https://github.com/Leeroo-AI/mergoo)

