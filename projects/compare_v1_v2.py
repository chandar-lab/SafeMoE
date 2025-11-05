"""
Comparison script to show the difference between V1 and V2 approaches.

This script demonstrates:
1. V1: Current working version with reinitialization workaround
2. V2: Improved version with proper weight saving
"""

import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open


def compare_checkpoints():
    """Compare V1 and V2 checkpoints to show the difference."""
    
    print("="*80)
    print("COMPARING V1 (CURRENT) vs V2 (IMPROVED)")
    print("="*80)
    
    v1_checkpoint = "data/test_qwen3_moe_checkpoint"
    v2_checkpoint = "data/test_qwen3_moe_checkpoint_v2"
    
    # ============ ANALYZE V1 CHECKPOINT ============
    print("\n" + "="*80)
    print("ANALYZING V1 CHECKPOINT (Current Working Version)")
    print("="*80)
    
    try:
        print(f"\nCheckpoint path: {v1_checkpoint}")
        
        # Load V1 checkpoint
        with safe_open(f"{v1_checkpoint}/model.safetensors", framework="pt") as f:
            v1_keys = list(f.keys())
            v1_norm_keys = [k for k in v1_keys if 'norm' in k.lower() and 'weight' in k]
            v1_gate_keys = [k for k in v1_keys if 'gate.weight' in k and 'mlp' in k]
            v1_lora_keys = [k for k in v1_keys if 'lora' in k]
        
        print(f"\nV1 Checkpoint Statistics:")
        print(f"  Total weights: {len(v1_keys)}")
        print(f"  Normalization weights: {len(v1_norm_keys)}")
        print(f"  Gate weights: {len(v1_gate_keys)}")
        print(f"  LoRA weights: {len(v1_lora_keys)}")
        
        if len(v1_norm_keys) == 0:
            print("\n⚠️  WARNING: V1 checkpoint has NO normalization weights!")
            print("This is why V1 needs reinitialization after loading.")
        
        # Load model from V1 checkpoint
        print("\nLoading model from V1 checkpoint...")
        model_v1 = AutoModelForCausalLM.from_pretrained(
            v1_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        # Check normalization weights BEFORE reinitialization
        print("\nChecking normalization weights BEFORE reinitialization:")
        v1_norm_params_before = []
        for name, param in model_v1.named_parameters():
            if 'norm' in name.lower() and 'weight' in name:
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                mean_val = param.mean().item() if not has_nan else float('nan')
                v1_norm_params_before.append((name, has_nan, has_inf, mean_val))
        
        nan_count = sum(1 for _, has_nan, _, _ in v1_norm_params_before if has_nan)
        inf_count = sum(1 for _, _, has_inf, _ in v1_norm_params_before if has_inf)
        
        print(f"  Total normalization weights: {len(v1_norm_params_before)}")
        print(f"  Weights with NaN: {nan_count}")
        print(f"  Weights with Inf: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print("\n❌ V1 has corrupted normalization weights after loading!")
            print("This is why V1 needs the reinitialization workaround.")
        
    except Exception as e:
        print(f"\n⚠️  Could not analyze V1 checkpoint: {e}")
        print("V1 checkpoint may not exist yet. Run test_qwen3_moe_flow.py first.")
    
    # ============ ANALYZE V2 CHECKPOINT ============
    print("\n" + "="*80)
    print("ANALYZING V2 CHECKPOINT (Improved Version)")
    print("="*80)
    
    try:
        print(f"\nCheckpoint path: {v2_checkpoint}")
        
        # Load V2 checkpoint
        with safe_open(f"{v2_checkpoint}/model.safetensors", framework="pt") as f:
            v2_keys = list(f.keys())
            v2_norm_keys = [k for k in v2_keys if 'norm' in k.lower() and 'weight' in k]
            v2_gate_keys = [k for k in v2_keys if 'gate.weight' in k and 'mlp' in k]
            v2_lora_keys = [k for k in v2_keys if 'lora' in k]
        
        print(f"\nV2 Checkpoint Statistics:")
        print(f"  Total weights: {len(v2_keys)}")
        print(f"  Normalization weights: {len(v2_norm_keys)}")
        print(f"  Gate weights: {len(v2_gate_keys)}")
        print(f"  LoRA weights: {len(v2_lora_keys)}")
        
        if len(v2_norm_keys) > 0:
            print("\n✅ V2 checkpoint HAS normalization weights!")
            print("This is why V2 doesn't need reinitialization.")
        
        # Load model from V2 checkpoint
        print("\nLoading model from V2 checkpoint...")
        model_v2 = AutoModelForCausalLM.from_pretrained(
            v2_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        # Check normalization weights (no reinitialization needed)
        print("\nChecking normalization weights (no reinitialization):")
        v2_norm_params = []
        for name, param in model_v2.named_parameters():
            if 'norm' in name.lower() and 'weight' in name:
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                mean_val = param.mean().item() if not has_nan else float('nan')
                v2_norm_params.append((name, has_nan, has_inf, mean_val))
        
        nan_count = sum(1 for _, has_nan, _, _ in v2_norm_params if has_nan)
        inf_count = sum(1 for _, _, has_inf, _ in v2_norm_params if has_inf)
        
        print(f"  Total normalization weights: {len(v2_norm_params)}")
        print(f"  Weights with NaN: {nan_count}")
        print(f"  Weights with Inf: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("\n✅ V2 has clean normalization weights after loading!")
            print("No reinitialization needed - weights loaded correctly.")
            
            # Show sample values
            print("\nSample normalization weight values:")
            for name, _, _, mean_val in v2_norm_params[:5]:
                print(f"  {name}: mean={mean_val:.4f}")
        
    except Exception as e:
        print(f"\n⚠️  Could not analyze V2 checkpoint: {e}")
        print("V2 checkpoint may not exist yet. Run test_qwen3_moe_flow_v2.py first.")
    
    # ============ COMPARISON SUMMARY ============
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\n📊 Key Differences:")
    print("\n1. Checkpoint Contents:")
    print("   V1: Missing normalization weights (only adapters + gates)")
    print("   V2: Contains ALL weights (base model + adapters + gates)")
    
    print("\n2. Loading Behavior:")
    print("   V1: PyTorch randomly initializes missing weights → NaN in bf16")
    print("   V2: PyTorch loads all weights from checkpoint → No NaN")
    
    print("\n3. Workaround Needed:")
    print("   V1: YES - Must reinitialize all norm weights to 1.0")
    print("   V2: NO - All weights load correctly")
    
    print("\n4. Model Quality:")
    print("   V1: Loses pretrained normalization behavior")
    print("   V2: Preserves full pretrained model quality")
    
    print("\n5. Code Complexity:")
    print("   V1: Higher - Requires manual reinitialization")
    print("   V2: Lower - Clean workflow")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\n✅ Use V2 for production training")
    print("   - Preserves full pretrained model quality")
    print("   - Cleaner code without workarounds")
    print("   - Follows standard practices")
    
    print("\n⚠️  Keep V1 as backup")
    print("   - Currently working")
    print("   - Can be used if V2 has issues")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    compare_checkpoints()

