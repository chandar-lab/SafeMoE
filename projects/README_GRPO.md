# GRPO Training for MoE Routers

This document explains how to use the GRPO (Group Relative Policy Optimization) training implementation in `exmp_1_grpo.py`.

## Overview

The script supports two training modes:

1. **SFT (Supervised Fine-Tuning)** - Traditional training mode
2. **GRPO** - Reinforcement learning with DeepSeek-style reward function

## GRPO Reward Function

The reward function consists of two components, similar to DeepSeek's approach:

### 1. Safety/Correctness Component (Using Llama Guard 3)
- **Informative/Safe responses**: +1.0
- **Unsafe responses**: -1.0
- **Refusal responses**: 0.0

Uses the actual `meta-llama/Llama-Guard-3-8B` model for safety assessment, with automatic fallback to heuristic-based checking if the model fails to load.

### 2. Format Quality Component [0.0, 0.1]
Rewards well-structured responses based on:
- **Expected format adherence**:
  - Full format (`### Question: ... ### Answer: ...`): +0.04
  - Partial format (`### Answer: ...`): +0.02
- **General quality** (up to +0.06):
  - Proper length (not too short/long)
  - Complete sentences with punctuation
  - Proper capitalization
  - Coherent structure

**Total reward range**: [-1.0, +1.1]

### Example Scoring Results:
- **Perfect format + safe content**: +1.100 (maximum reward)
- **Partial format + safe content**: +1.085
- **No format markers + safe content**: +1.075
- **Perfect format + unsafe content**: -0.900 (penalized for safety)
- **Refusal + partial format**: +0.085 (neutral safety + format bonus)
- **Short, poor format**: +1.000 (safe but no format bonus)

## Configuration

### Using Existing Config with GRPO Flag

Edit `conf/config.yaml` and set:

```yaml
grpo:
  enabled: true  # Enable GRPO training
  per_device_train_batch_size: 1
  learning_rate: 5e-6  # Lower than SFT
  num_train_epochs: 3  # Fewer than SFT
  num_generations: 4  # Completions per prompt
  max_completion_length: 256
  temperature: 0.7
  beta: 0.0  # KL coefficient

llama_guard:
  enabled: true  # Use actual Llama Guard model
  model_id: "meta-llama/Llama-Guard-3-8B"
  device_map: "auto"
  torch_dtype: "bfloat16"
```

### Using Dedicated GRPO Config

Use the pre-configured GRPO config file:

```bash
python exmp_1_grpo.py --config-name=config_grpo
```

## Usage Examples

### 1. SFT Training (Default)
```bash
python exmp_1_grpo.py
```

### 2. GRPO Training with Default Config
```bash
# Edit config.yaml to set grpo.enabled=true, then:
python exmp_1_grpo.py
```

### 3. GRPO Training with Dedicated Config
```bash
python exmp_1_grpo.py --config-name=config_grpo
```

### 4. Override GRPO Parameters
```bash
python exmp_1_grpo.py --config-name=config_grpo grpo.num_generations=8 grpo.temperature=0.8
```

## Testing the Reward Function

Test the reward function with sample completions:

```bash
python test_grpo_reward.py
```

This will show how different types of responses are scored by the reward function.

## Key Differences: SFT vs GRPO

| Aspect | SFT | GRPO |
|--------|-----|------|
| **Training Type** | Supervised | Reinforcement Learning |
| **Data Format** | Question + Answer pairs | Prompts only |
| **Learning Rate** | 1e-4 | 5e-6 (lower) |
| **Epochs** | 10 | 3 (fewer) |
| **Evaluation** | Standard metrics | Reward-based |
| **Generation** | Not required | Multiple per prompt |

## GRPO-Specific Parameters

- `num_generations`: Number of completions generated per prompt (default: 4)
- `max_completion_length`: Maximum tokens in generated responses (default: 256)
- `temperature`: Generation randomness (0.0=deterministic, 1.0=random)
- `beta`: KL divergence penalty coefficient (0.0=no penalty)

## Monitoring Training

The script logs GRPO-specific metrics to Weights & Biases:
- Reward statistics (mean, std)
- Generation statistics (length, completion rate)
- Training mode and parameters

## Hardware Requirements

GRPO training requires more memory than SFT due to:
- Multiple generation passes
- Reward computation
- Policy comparison

Consider using:
- Larger batch sizes if memory allows
- Gradient accumulation for effective batch size
- Mixed precision training (bf16)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `num_generations` or `max_completion_length`
2. **Slow Training**: Enable bf16, increase batch size, use gradient accumulation
3. **Poor Rewards**: Check reward function logic, adjust temperature
4. **No Improvement**: Increase learning rate, check data quality

### Debug Mode

Add debug prints to reward function:

```python
def deepseek_style_reward_function(completions, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        safety_score = llama_guard_judge(completion)
        format_score = calculate_format_score(completion)
        total_reward = safety_score + format_score
        print(f"Sample {i}: Safety={safety_score}, Format={format_score:.3f}, Total={total_reward:.3f}")
        rewards.append(total_reward)
    return rewards
```

## Advanced Usage

### Custom Reward Functions

Replace `deepseek_style_reward_function` with your own:

```python
def custom_reward_function(completions, **kwargs):
    # Your custom logic here
    return [score_for_completion(comp) for comp in completions]
```

### Multi-GPU Training

Use accelerate for distributed training:

```bash
accelerate launch --multi_gpu exmp_1_grpo.py --config-name=config_grpo
```
