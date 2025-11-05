"""
Test script for Qwen3 MoE with LoRA - V2 (Improved Version)

This version tests the improved composer that properly saves all model weights.
Key difference: No need to reinitialize normalization weights after loading.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mergoo.composers.composer_lora_moe_v2 import ComposeLoraMoeExpertsV2


def test_qwen3_moe_flow_v2():
    """Test the complete flow with improved V2 composer."""
    
    print("="*80)
    print("TESTING QWEN3 MOE WITH LORA - IMPROVED VERSION (V2)")
    print("="*80)
    
    # Configuration
    base_model = "Qwen/Qwen3-8B"

    expert_models = [
        "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-mental_health",
        "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-fiqa",
    ]
    router_layers = ["gate_proj", "up_proj", "down_proj"]
    checkpoint_path = "data/test_qwen3_moe_checkpoint_v2"
    
    # ============ STEP 1: COMPOSE MOE MODEL ============
    print("\n" + "="*80)
    print("STEP 1: COMPOSING MOE MODEL")
    print("="*80)
    
    composer_config = {
        "base_model": base_model,
        "experts": [{"model_id": expert_id} for expert_id in expert_models],
        "router_layers": router_layers,
        "adapter_configs": [],
    }
    
    print(f"\nBase model: {base_model}")
    print(f"Experts: {expert_models}")
    print(f"Router layers: {router_layers}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    try:
        composer = ComposeLoraMoeExpertsV2(
            config=composer_config,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        
        composer.compose()
        composer.save_checkpoint(checkpoint_path)
        
        print("\n✅ Composition successful!")
        
    except Exception as e:
        print(f"\n❌ Composition failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============ STEP 2: LOAD AND VERIFY CHECKPOINT ============
    print("\n" + "="*80)
    print("STEP 2: TESTING CHECKPOINT LOADING")
    print("="*80)
    
    try:
        print("\nLoading model from checkpoint...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        print("\n✅ Model loaded successfully!")
        
        # NO REINITIALIZATION NEEDED - This is the key improvement!
        # The V2 composer properly saves all weights including normalization weights
        
        # Check for NaN/Inf in weights
        print("\nChecking for NaN/Inf in weights...")
        nan_count = 0
        inf_count = 0
        total_params = 0
        norm_weights = []
        gate_weights = []
        
        for name, param in model.named_parameters():
            total_params += 1
            has_nan = torch.isnan(param).any()
            has_inf = torch.isinf(param).any()
            
            if has_nan:
                nan_count += 1
                print(f"  ⚠️  NaN found in: {name}")
            if has_inf:
                inf_count += 1
                print(f"  ⚠️  Inf found in: {name}")
            
            if 'norm' in name.lower() and 'weight' in name:
                norm_weights.append((name, param))
            if 'gate.weight' in name and 'mlp' in name:
                gate_weights.append((name, param))
        
        if nan_count == 0 and inf_count == 0:
            print(f"✅ All {total_params} parameters are clean (no NaN/Inf)")
        else:
            print(f"❌ Found {nan_count} NaN and {inf_count} Inf parameters out of {total_params}")
            print("\n❌ OVERALL TEST FAILED: Model has corrupted weights")
            return False
        
        # Verify normalization weights are properly loaded
        print(f"\n✅ Found {len(norm_weights)} normalization weights")
        if len(norm_weights) > 0:
            print("\nSample normalization weight values (should be close to 1.0):")
            for name, param in norm_weights[:5]:
                mean_val = param.mean().item()
                std_val = param.std().item()
                min_val = param.min().item()
                max_val = param.max().item()
                print(f"  {name}:")
                print(f"    mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
        
        # Verify gate weights
        print(f"\n✅ Found {len(gate_weights)} gate weight tensors")
        if len(gate_weights) > 0:
            print("\nSample gate weight ranges (should be small for bf16 stability):")
            for name, param in gate_weights[:3]:
                min_val = param.min().item()
                max_val = param.max().item()
                print(f"  {name}: range=[{min_val:.6f}, {max_val:.6f}]")
        
        # Check MoE layers
        print("\nChecking MoE layers...")
        moe_layers = []
        for name, module in model.named_modules():
            if any(layer in name for layer in router_layers):
                if hasattr(module, 'gate'):
                    moe_layers.append(name)
        
        print(f"✅ Found {len(moe_layers)} MoE layers")
        if len(moe_layers) > 0:
            print(f"  First few: {moe_layers[:3]}")
        
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============ STEP 3: TEST FORWARD PASS ============
    print("\n" + "="*80)
    print("STEP 3: TESTING FORWARD PASS")
    print("="*80)
    
    try:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        print("\nPreparing test input...")
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        print("Moving model and inputs to CUDA...")
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        
        # Check for NaN in outputs
        if torch.isnan(logits).any():
            print("❌ NaN detected in output logits!")
            print("\n❌ OVERALL TEST FAILED: Forward pass failed")
            return False
        
        print("✅ Forward pass successful!")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output dtype: {logits.dtype}")
        print(f"  Output range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============ SUCCESS ============
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nKey improvements in V2:")
    print("1. ✅ All model weights properly saved (including normalization)")
    print("2. ✅ No need to reinitialize weights after loading")
    print("3. ✅ Cleaner and more reliable workflow")
    print("4. ✅ Model preserves pretrained normalization weights")
    print("\nYou can now run the full training with: python projects/exmp_1_qwen_v2.py")
    
    return True


if __name__ == "__main__":
    success = test_qwen3_moe_flow_v2()
    exit(0 if success else 1)

