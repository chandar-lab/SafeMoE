# Alpha MoE Training

This directory contains the implementation for training alpha coefficients in Mixture of Experts (MoE) models with LoRA adapters.

## Overview

The alpha MoE training introduces trainable alpha coefficients (α) for each layer of the MoE model. Instead of directly merging LoRA weights with the base model, we use:

```
final_weight = α * w_base + (1-α) * δw_moe
```

Where:
- `α` is a trainable coefficient (0 ≤ α ≤ 1)
- `w_base` is the base model weight
- `δw_moe` is the MoE LoRA delta weight

## Key Features

- **Trainable Alpha Coefficients**: Each MoE layer gets its own set of alpha parameters
- **Mixed Dataset Training**: Combines safe evaluation samples with Alpaca dataset
- **Frozen Base Weights**: Only alpha parameters are trained, everything else is frozen
- **Layer-wise Control**: Fine-grained control over base vs. MoE weight contribution

## Files

- `train_alpha_moe.py`: Main training script
- `test_alpha_moe.py`: Test script to validate the implementation
- `run_alpha_training.sh`: Shell script to run training with default parameters
- `README_alpha_moe.md`: This documentation file

## Usage

### Quick Start

```bash
# Make the script executable (if not already)
chmod +x projects/run_alpha_training.sh

# Run with default parameters
./projects/run_alpha_training.sh

# Or specify a custom MoE model path
./projects/run_alpha_training.sh projects/data/your_moe_model
```

### Manual Training

```bash
python projects/train_alpha_moe.py \
    --moe_model_path "projects/data/mistral_lora_moe_all_XL_22apater_top2" \
    --safe_data_path "projects/evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json" \
    --output_dir "projects/data/alpha_moe_trained" \
    --learning_rate 1e-3 \
    --num_epochs 5 \
    --batch_size 2 \
    --alpaca_percentage 0.2
```

### Testing

Before running training, test the implementation:

```bash
python projects/test_alpha_moe.py
```

## Parameters

- `--moe_model_path`: Path to the pre-trained MoE model checkpoint
- `--safe_data_path`: Path to the safe evaluation data JSON file
- `--output_dir`: Directory to save the trained model and alpha coefficients
- `--learning_rate`: Learning rate for alpha parameters (default: 1e-3)
- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 2)
- `--alpaca_percentage`: Percentage of Alpaca dataset to include (default: 0.2)

## Dataset

The training uses a mixed dataset consisting of:

1. **Safe Samples**: From the evaluation data where `"is_safe": true`
2. **Alpaca Subset**: 20% (configurable) of the tatsu-lab/alpaca dataset
   - Format: `instruction + input` → `output`

## Output

The training produces:

1. `alpha_coefficients.json`: Final alpha values for each layer and adapter
2. `training_config.json`: Training configuration and statistics
3. Model checkpoint with trained alpha parameters

## Implementation Details

### AlphaMoELinear Class

Enhances the existing MoE linear layers by:
- Adding trainable alpha coefficients for each adapter
- Modifying the forward pass to apply alpha weighting
- Preserving the original MoE routing mechanism

### AlphaMoEModel Class

Wraps the base MoE model by:
- Automatically detecting and replacing MoE layers
- Freezing all parameters except alpha coefficients
- Providing utilities to monitor alpha values

## Example Alpha Values

After training, you might see alpha values like:

```json
{
  "model.layers.0.mlp.gate_proj": {
    "0": 0.7234,
    "1": 0.4567
  },
  "model.layers.0.mlp.up_proj": {
    "0": 0.8123,
    "1": 0.3456
  }
}
```

Where higher alpha values favor the base model weights, and lower values favor the MoE LoRA weights.

## Requirements

- PyTorch
- Transformers
- TRL (for SFTTrainer)
- Datasets
- The mergoo library (from this repository)

## Troubleshooting

1. **No MoE layers found**: Ensure the model path points to a valid MoE model with LoRA adapters
2. **No trainable parameters**: Check if the model has the expected MoE structure
3. **CUDA out of memory**: Reduce batch size or use gradient accumulation
4. **Dataset loading errors**: Verify the safe data file exists and has the correct format

## Next Steps

After training, you can:
1. Evaluate the model with the new alpha coefficients
2. Analyze which layers learned to favor base vs. MoE weights
3. Use the alpha values to understand model behavior
4. Fine-tune alpha values further on different datasets
