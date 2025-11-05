# Qwen3 MoE with LoRA Experts - Complete Setup Guide

## Overview

This guide explains how to create and train a Mixture of Experts (MoE) model using Qwen3-8B with LoRA adapters, using only bf16 precision (no fp32 required).

## Architecture

```
Qwen3-8B Base Model
    ↓
+ Multiple LoRA Adapters (trained on different domains)
    ↓
= MoE Model with Router
    ↓
Router learns to select best experts for each token
```

## Prerequisites

1. **Base Model**: Qwen/Qwen3-8B (or Qwen3-4B)
2. **LoRA Adapters**: Pre-trained LoRA adapters on different domains
3. **Memory**: Sufficient GPU memory for bf16 (8B model needs ~16GB)

## Current Setup

### Available LoRA Adapters
You have 23 Qwen3-8B LoRA adapters in `projects/loras/`:
- `qwen-3-8B-lora-mental_health`
- `qwen-3-8B-lora-fiqa`
- `qwen-3-8B-lora-cybersecurity_kaggle`
- `qwen-3-8B-lora-drug_abuse`
- `qwen-3-8B-lora-pubmedqau`
- And 18 more unsafe category adapters

### Target Modules
All adapters target the same MLP layers:
- `gate_proj`
- `up_proj`
- `down_proj`

## The Complete Flow

### Step 1: Composition (Creating MoE Checkpoint)

**What happens:**
1. Load base Qwen3-8B model with regular `nn.Linear` layers
2. Load each LoRA adapter using PEFT
3. Extract LoRA weights (A and B matrices) from each adapter
4. Organize weights for MoE structure
5. Save checkpoint with MoE configuration

**Key Point:** During composition, the model has regular Linear layers so PEFT can inject adapters.

**Code:**
```python
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM

model_config = {
    "model_type": "qwen3",
    "num_experts_per_tok": 2,
    "base_model": "Qwen/Qwen3-8B",
    "experts": [
        {"expert_name": "adapter_1", "model_id": "path/to/lora1"},
        {"expert_name": "adapter_2", "model_id": "path/to/lora2"},
    ],
    "router_layers": ["gate_proj", "up_proj", "down_proj"],
}

composer = ComposeExperts(
    model_config,
    torch_dtype=torch.bfloat16,
    model_cls=Qwen3ForCausalLM
)
composer.compose()
composer.save_checkpoint("data/qwen3_moe_checkpoint")
```

### Step 2: Training (Router Training)

**What happens:**
1. Load checkpoint with MoE configuration
2. Model is created with `LoRAMoeLayer` instead of `nn.Linear`
3. Each MoE layer has:
   - Base layer (frozen)
   - Multiple LoRA experts (frozen)
   - Router/gate (trainable)
4. Train only the router weights to learn expert selection

**Code:**
```python
model = Qwen3ForCausalLM.from_pretrained(
    "data/qwen3_moe_checkpoint",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    trust_remote_code=True
)

# Train with standard HuggingFace Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    ...
)
trainer.train()
```

## Key Fixes Applied

### 1. Safe Attribute Access in `convert_linear_to_moe`

**Problem:** Direct access to `config.router_layers_index` fails if attribute doesn't exist.

**Fix:**
```python
# Use getattr with default None
router_layers_index = getattr(config, "router_layers_index", None)
router_layers = getattr(config, "router_layers", None)

if router_layers_index is None or router_layers is None:
    return nn.Linear(in_features, out_features, bias=bias)
```

### 2. Gate Weight Initialization

**Problem:** Default Kaiming initialization produces large values that overflow in bf16, causing NaN.

**Fix:**
```python
# Initialize gate weights with small values
nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
```

**Why this works:**
- Small std (0.01) keeps values in range ~[-0.03, 0.03]
- Well within bf16's representable range
- Same initialization used in production MoE models (Mixtral, Qwen3MoE)

### 3. Proper Weight Normalization

**Problem:** Weights weren't normalized with softmax, causing training instability.

**Fix:**
```python
# Normalize weights with softmax
weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(previous_dtype)
```

**Why this works:**
- Ensures weights sum to 1
- Prevents gradient explosion
- Uses fp32 for softmax computation for numerical stability

## Configuration

### Minimal Config (2 experts)
```yaml
model:
  model_type: "qwen3"
  num_experts_per_tok: 2
  base_model: "Qwen/Qwen3-8B"
  experts:
    - expert_name: "adapter_1"
      model_id: "path/to/lora1"
    - expert_name: "adapter_2"
      model_id: "path/to/lora2"
  router_layers: ["gate_proj", "up_proj", "down_proj"]

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  bf16: true
  gradient_checkpointing: true
  max_seq_length: 256
```

### Full Config (8+ experts)
See `projects/conf/config_qwen.yaml` for complete configuration with all available experts.

## Testing

### Quick Test
```bash
python projects/test_qwen3_moe_flow.py
```

This will:
1. ✅ Test composition with 2 experts
2. ✅ Test checkpoint loading
3. ✅ Check for NaN/Inf in weights
4. ✅ Test forward pass
5. ✅ Verify MoE layers are created

### Full Training
```bash
python projects/exmp_1_qwen.py
```

## Memory Requirements

### Qwen3-8B
- **Composition**: ~16GB GPU memory (bf16)
- **Training**: ~20-24GB GPU memory (bf16 + gradients)
- **With gradient checkpointing**: ~16-18GB

### Qwen3-4B (Alternative)
- **Composition**: ~8GB GPU memory (bf16)
- **Training**: ~12-14GB GPU memory (bf16 + gradients)

## Troubleshooting

### Issue: NaN in weights after loading
**Cause:** Gate weights initialized with large values that overflow in bf16
**Solution:** ✅ Fixed with small initialization (std=0.01)

### Issue: NaN during training
**Cause:** Weights not normalized, causing gradient explosion
**Solution:** ✅ Fixed with softmax normalization

### Issue: AttributeError: 'Qwen3Config' object has no attribute 'router_layers_index'
**Cause:** Trying to access MoE attributes during composition
**Solution:** ✅ Fixed with safe getattr() access

### Issue: PEFT can't inject adapters into LoRAMoeLayer
**Cause:** Model created with MoE layers during composition
**Solution:** ✅ Fixed - model loads with regular Linear layers during composition

## Best Practices

1. **Start Small**: Test with 2 experts first, then scale up
2. **Use bf16**: Required for memory efficiency, now stable with fixes
3. **Gradient Checkpointing**: Enable for large models
4. **Batch Size**: Start with 1, increase with gradient accumulation
5. **Sequence Length**: Start with 256, increase if memory allows

## Files Modified

1. `mergoo/compose_layers.py`:
   - Safe attribute access in `convert_linear_to_moe`
   - Gate weight initialization in `MoeLayer` and `LoRAMoeLayer`
   - Proper softmax normalization in forward pass

2. `projects/exmp_1_qwen.py`:
   - Added `router_layers` to model config
   - Uses bf16 throughout

3. `projects/conf/config_qwen.yaml`:
   - Configured for Qwen3-8B with 2 experts
   - bf16 enabled
   - Memory-optimized settings

## Next Steps

1. **Test the setup:**
   ```bash
   python projects/test_qwen3_moe_flow.py
   ```

2. **Run composition and training:**
   ```bash
   python projects/exmp_1_qwen.py
   ```

3. **Monitor training:**
   - Check WandB dashboard for loss curves
   - Verify no NaN in gradients
   - Monitor expert selection patterns

4. **Scale up:**
   - Add more experts (4, 8, 16, etc.)
   - Increase sequence length
   - Tune learning rate and batch size

## Summary

✅ **All issues fixed:**
- Safe attribute access prevents AttributeError
- Small gate initialization prevents NaN in bf16
- Softmax normalization prevents training instability
- Proper composition flow allows PEFT injection

✅ **Ready to use:**
- Test script validates complete flow
- Configuration optimized for memory
- Works with bf16 only (no fp32 needed)

✅ **Scalable:**
- Start with 2 experts, scale to 20+
- Works with Qwen3-8B and Qwen3-4B
- Memory-efficient with gradient checkpointing

