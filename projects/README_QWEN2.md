# Qwen2.5-1.5B MoE Training with Mergoo

This guide explains how to use the Qwen2.5-1.5B model with Mergoo's MoE (Mixture of Experts) functionality.

## Overview

The Qwen2 implementation (`mergoo/models/modeling_qwen2.py`) supports:
- **Composer LoRA MoE**: Use LoRA adapters as experts
- **Composer MoE**: Use full model experts
- **Compose Layers**: Selective MoE application to specific layers
- **Router Training**: Train only the routing/gating layers while keeping experts frozen

## Files

- **`exmp_qwen2.py`**: Main training script for Qwen2 MoE
- **`conf/config_qwen2.yaml`**: Configuration file for Qwen2 MoE training

## Quick Start

### 1. Prepare Your LoRA Adapters

First, train or obtain LoRA adapters for Qwen2.5-1.5B on your domain-specific datasets. Each adapter will become an expert in the MoE model.

### 2. Update Configuration

Edit `projects/conf/config_qwen2.yaml`:

```yaml
model:
  model_type: "qwen2"
  num_experts_per_tok: 2  # Number of experts to activate per token (top-k)
  base_model: "Qwen/Qwen2.5-1.5B"
  experts:
    - expert_name: "adapter_1"
      model_id: "/path/to/your/qwen2-lora-adapter-1"
    - expert_name: "adapter_2"
      model_id: "/path/to/your/qwen2-lora-adapter-2"
    # Add more experts as needed
  
  # Choose which layers to apply MoE
  # For MLP layers:
  router_layers: ["gate_proj", "up_proj", "down_proj"]
  # For attention layers:
  # router_layers: ["q_proj", "k_proj", "v_proj", "o_proj"]
  # For both:
  # router_layers: ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]

data:
  dataset_name: "/path/to/your/training/dataset.json"
```

### 3. Run Training

```bash
# Load required modules (if on a cluster)
module load python/3.10
module load cuda/12.1.1

# Run training
cd projects
python exmp_qwen2.py
```

## Configuration Options

### Model Configuration

- **`model_type`**: Must be `"qwen2"` for Qwen2 models
- **`num_experts_per_tok`**: Number of experts to activate per token (typically 1-2)
- **`base_model`**: Base Qwen2 model to use (e.g., `"Qwen/Qwen2.5-1.5B"`)
- **`experts`**: List of LoRA adapters to use as experts
- **`router_layers`**: Which layers to convert to MoE

### Router Layers Options

You can apply MoE to different parts of the model:

1. **MLP Layers Only** (recommended for most cases):
   ```yaml
   router_layers: ["gate_proj", "up_proj", "down_proj"]
   ```

2. **Attention Layers Only**:
   ```yaml
   router_layers: ["q_proj", "k_proj", "v_proj", "o_proj"]
   ```

3. **Both MLP and Attention**:
   ```yaml
   router_layers: ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
   ```

### Training Configuration

- **`per_device_train_batch_size`**: Batch size per device (default: 1)
- **`learning_rate`**: Learning rate for router training (default: 1e-4)
- **`num_train_epochs`**: Number of training epochs (default: 10)
- **`gradient_accumulation_steps`**: Gradient accumulation steps (default: 4)
- **`max_seq_length`**: Maximum sequence length (default: 512)

## Dataset Format

Your training dataset should be a JSON file with the following format:

```json
[
  {
    "prompt": "What is artificial intelligence?",
    "response": "Artificial intelligence (AI) is..."
  },
  {
    "instruction": "Explain machine learning",
    "response": "Machine learning is a subset of AI..."
  }
]
```

The script supports both `"prompt"` and `"instruction"` as the question field.

## Advanced Usage

### Using DeepSpeed

Uncomment the DeepSpeed configuration in `config_qwen2.yaml`:

```yaml
deepspeed:
  enabled: true
  config_path: "conf/deepspeed_config.json"
```

### Selective Layer MoE

To apply MoE only to specific transformer layers, you can modify the config after loading:

```python
# In your training script
config.router_layers_index = [0, 1, 2]  # Apply MoE only to first 3 layers
```

### Training with Wandb

The script automatically logs to Weights & Biases if enabled:

```yaml
wandb:
  enabled: true
  project: "qwen2-moe-training"
  tags: ["qwen2", "moe", "your-experiment"]
```

## Model Architecture Details

### Qwen2 vs Mistral Differences

The Qwen2 implementation differs from Mistral in the following ways:

1. **Attention Bias**: Qwen2 uses `bias=True` for q_proj, k_proj, v_proj (Mistral uses `bias=False`)
2. **Output Projection**: Both use `bias=False` for o_proj
3. **Default Hidden Size**: Qwen2.5-1.5B uses hidden_size=1536, intermediate_size=8960

### MoE Integration

All linear layers in the model can be converted to MoE layers using the `convert_linear_to_moe` function:

- **MLP layers**: `gate_proj`, `up_proj`, `down_proj`
- **Attention layers**: `q_proj`, `k_proj`, `v_proj`, `o_proj`

The router (gating) network learns to select the best experts for each token.

## Troubleshooting

### Import Errors

If you encounter import errors, make sure you're loading the model directly:

```python
from mergoo.models.modeling_qwen2 import Qwen2ForCausalLM
```

### Memory Issues

If you run out of memory:
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Reduce `max_seq_length`
4. Use DeepSpeed with ZeRO optimization

### Cache API Errors

The implementation handles different transformers versions automatically. If you encounter cache-related errors, ensure you're using a compatible transformers version (4.36+).

## Example: Complete Training Pipeline

```bash
# 1. Prepare your LoRA adapters (train them first)
# 2. Update config_qwen2.yaml with your adapter paths and dataset
# 3. Run training

module load python/3.10
module load cuda/12.1.1

cd projects
python exmp_qwen2.py

# Monitor training with wandb
# The model checkpoint will be saved to data/qwen2_lora_moe_checkpoint
# Training outputs will be in data/output/qwen2_lora_moe_training
```

## Citation

If you use this implementation, please cite the original Qwen2 paper and the Mergoo framework.

## Support

For issues or questions:
1. Check the main Mergoo documentation
2. Review the Mistral example (`exmp_1.py`) for reference
3. Ensure your LoRA adapters are compatible with Qwen2.5-1.5B

