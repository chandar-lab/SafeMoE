# Alpha MoE Text Generation

This directory contains scripts for training and using Alpha MoE models with vector-based alpha coefficients.

## Files

- `train_alpha_moe_v2.py` - Training script for alpha coefficients (V2 with vector alphas)
- `generate_with_alpha_moe.py` - Text generation script using trained alpha model
- `example_alpha_generation.py` - Example usage script with multiple test prompts

## Key Features of V2

The V2 implementation introduces two major improvements:

1. **Single Alpha Over Mixture**: Instead of having separate alpha values for each expert, there's now one alpha that controls the mixing between base model and the entire expert mixture.

2. **Vector Alpha**: Alpha is now a vector (same dimensionality as output features) rather than a scalar, allowing fine-grained control per output dimension.

## Usage

### 1. Train Alpha Coefficients

First, train the alpha coefficients using your MoE model:

```bash
python projects/train_alpha_moe_v2.py \
    --moe_model_path /path/to/your/moe/model \
    --safe_data_path /path/to/safe/data.json \
    --output_dir /path/to/output/alpha_trained_model \
    --learning_rate 1e-4 \
    --num_epochs 10
```

This will create:
- `alpha_coefficients.json` - The trained alpha vectors
- `training_config.json` - Training configuration
- Model checkpoint files

### 2. Generate Text

#### Simple Generation
```bash
python projects/generate_with_alpha_moe.py \
    --model_path /path/to/output/alpha_trained_model \
    --prompt "What is machine learning?"
```

#### Interactive Mode
```bash
python projects/generate_with_alpha_moe.py \
    --model_path /path/to/output/alpha_trained_model \
    --interactive
```

#### Run Examples
```bash
python projects/example_alpha_generation.py
```

## Alpha Coefficient Interpretation

The alpha vector controls mixing between base model and expert mixture:

- **α = 1.0**: Use only base model output (ignore experts)
- **α = 0.0**: Use only expert mixture output (ignore base model)  
- **α = 0.5**: Equal mixing of base and expert outputs

Since alpha is now a vector, each output dimension can have different mixing ratios, allowing the model to learn which dimensions benefit more from expert knowledge vs base model knowledge.

## Example Output

When you run the generation script, you'll see:

```
=== Alpha MoE Text Generation Example ===

Loading tokenizer...
Loading trained alpha MoE model...
Replacing MoE layer: model.layers.0.mlp.gate_proj
Replacing MoE layer: model.layers.0.mlp.up_proj
...

Loaded Alpha Values:
  model.layers.0.mlp.gate_proj: vector shape=torch.Size([4096]), mean=0.4823, std=0.0156
  model.layers.0.mlp.up_proj: vector shape=torch.Size([11008]), mean=0.5012, std=0.0089
  ...

--- Example 1 ---
Prompt: What is the capital of France?
Response: The capital of France is Paris. Paris is located in the north-central part of France...
```

## Architecture Details

### AlphaMoELinear Layer
- Wraps original MoE layers
- Adds single alpha vector parameter per layer
- Applies element-wise mixing: `α * base + (1-α) * mixture`

### AlphaMoEModel
- Replaces all MoE layers with alpha-enhanced versions
- Manages alpha parameters across the entire model
- Supports loading/saving alpha coefficients

## Troubleshooting

1. **CUDA Out of Memory**: Use `--use_quantization` during training
2. **No MoE Layers Found**: Ensure your model has LoRA adapters with `active_adapters`
3. **Alpha File Not Found**: The script will use default values (0.5) if no alpha file is provided

## Comparison with V1

| Feature | V1 (Per-Expert) | V2 (Single Vector) |
|---------|-----------------|-------------------|
| Alpha Structure | Dict of scalars per expert | Single vector per layer |
| Parameters | `num_experts` scalars | `output_dim` vector elements |
| Mixing | Per-expert weighting | Global mixture weighting |
| Interpretability | Expert-specific control | Dimension-specific control |
| Memory | Lower (fewer params) | Higher (more params) |
| Flexibility | Expert granularity | Feature granularity |
