# V2 Quick Start Guide

## TL;DR

**Current V1 (Working but not ideal)**: Saves only adapter weights → Missing normalization weights → Must reinitialize to 1.0 → Loses pretrained quality

**Improved V2 (Proper solution)**: Saves ALL weights → All weights load correctly → No reinitialization needed → Preserves pretrained quality

## Quick Test

```bash
# Test V2 implementation
python projects/test_qwen3_moe_flow_v2.py

# Compare V1 vs V2
python projects/compare_v1_v2.py
```

## Files Created

### Core Implementation
- **`mergoo/composers/composer_lora_moe_v2.py`** - Improved composer that saves all weights
- **`projects/exmp_1_qwen_v2.py`** - Training script using V2
- **`projects/test_qwen3_moe_flow_v2.py`** - Test script for V2

### Documentation
- **`IMPROVED_SOLUTION_V2.md`** - Detailed explanation of V1 vs V2
- **`projects/compare_v1_v2.py`** - Script to compare V1 and V2 checkpoints
- **`V2_QUICK_START.md`** - This file

## Key Changes in V2

### 1. Composer (`composer_lora_moe_v2.py`)

```python
# V1 (Current - Problematic)
def save_checkpoint(self, checkpoint_path):
    model.save_pretrained(checkpoint_path)  # Only saves adapters!

# V2 (Improved - Correct)
def save_checkpoint(self, checkpoint_path):
    state_dict = model.state_dict()  # Get FULL state dict
    save_file(state_dict, f"{checkpoint_path}/model.safetensors")  # Save all weights
```

### 2. Training Script (`exmp_1_qwen_v2.py`)

```python
# V1 (Current - Needs workaround)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
# Must reinitialize norm weights!
for name, param in model.named_parameters():
    if 'norm' in name.lower():
        param.fill_(1.0)

# V2 (Improved - Clean)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
# All weights load correctly - no reinitialization needed!
```

## Why V2 Is Better

| Aspect | V1 | V2 |
|--------|----|----|
| Saves all weights | ❌ | ✅ |
| Preserves pretrained norms | ❌ | ✅ |
| Needs reinitialization | ❌ | ✅ |
| Standard practice | ❌ | ✅ |
| Model quality | ⚠️ | ✅ |

## Testing V2

### Step 1: Run Test Script

```bash
python projects/test_qwen3_moe_flow_v2.py
```

Expected output:
```
✅ ALL TESTS PASSED!

Key improvements in V2:
1. ✅ All model weights properly saved (including normalization)
2. ✅ No need to reinitialize weights after loading
3. ✅ Cleaner and more reliable workflow
4. ✅ Model preserves pretrained normalization weights
```

### Step 2: Compare with V1

```bash
python projects/compare_v1_v2.py
```

This will show you:
- V1 checkpoint: Missing normalization weights
- V2 checkpoint: Contains all weights
- Side-by-side comparison

### Step 3: Run Training

```bash
python projects/exmp_1_qwen_v2.py
```

## Answering Your Questions

### Q1: Does reinitializing normalization weights corrupt the model?

**Short answer**: It doesn't completely corrupt it, but it does degrade quality.

**Detailed answer**:
- Normalization weights in pretrained models have learned values (usually close to 1.0 but not exactly)
- These learned values encode important information about the data distribution
- Resetting them to exactly 1.0 loses this information
- The model still works, but may have:
  - Slower convergence during training
  - Lower final performance
  - Less stable training dynamics

**V2 solution**: Preserves the original learned values → Better quality

### Q2: Does it affect model performance?

**Yes, V1 likely has degraded performance compared to V2.**

**Why V1 is worse**:
- Loses pretrained normalization behavior
- May need more training to recover
- Final model may not reach the same quality

**Why V2 is better**:
- Preserves all pretrained knowledge
- Faster convergence
- Better final performance

**Recommendation**: Use V2 for production training to get the best results.

### Q3: Is this what others do?

**No, V1 is a workaround. V2 is the standard practice.**

**What the community does**:
1. **For adapter-only saving** (fine-tuning): Use `save_pretrained()` - saves only adapters
2. **For full model saving** (composition/merging): Extract `state_dict` and save manually - saves everything

**V1**: Uses approach #1 when it should use approach #2
**V2**: Uses approach #2 correctly

**Examples from popular libraries**:
- PEFT: Uses `save_pretrained()` for adapter-only saving
- Mergekit: Extracts full state_dict for model merging
- LLaMA-Factory: Extracts full state_dict for model composition

## Migration Checklist

- [ ] Test V2: `python projects/test_qwen3_moe_flow_v2.py`
- [ ] Compare: `python projects/compare_v1_v2.py`
- [ ] Verify: Check that all tests pass
- [ ] Train small model: Test V2 on small dataset
- [ ] Compare results: V1 vs V2 performance
- [ ] Switch to V2: Use for production training

## Troubleshooting

### If V2 test fails

1. Check that you have `safetensors` installed:
   ```bash
   pip install safetensors
   ```

2. Check that you have enough disk space for full checkpoint

3. Check error messages - they should be informative

### If V2 checkpoint is too large

V2 checkpoints are larger because they contain all weights:
- V1: ~500MB (adapters + gates only)
- V2: ~8GB (full model for Qwen3-8B)

This is expected and correct. The full model needs to be saved.

### If you want to keep using V1

V1 is working and you can keep using it if:
- You're okay with potentially degraded performance
- You don't have disk space for full checkpoints
- You're doing quick experiments

But for production training, V2 is recommended.

## Summary

**V1 (Current)**:
- ✅ Works
- ❌ Loses pretrained normalization
- ❌ Requires workaround
- ⚠️ May have degraded performance

**V2 (Improved)**:
- ✅ Works
- ✅ Preserves pretrained normalization
- ✅ Clean workflow
- ✅ Full performance

**Recommendation**: Test V2 and migrate to it for better model quality.

