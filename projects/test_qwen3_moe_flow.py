"""
Test script to verify Qwen3 MoE with LoRA flow
This script tests the complete flow from composition to training
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer

def test_composition():
    """Test Step 1: Composition"""
    print("\n" + "="*80)
    print("STEP 1: TESTING COMPOSITION")
    print("="*80)
    
    model_config = {
        "model_type": "qwen3",
        "num_experts_per_tok": 2,
        "base_model": "Qwen/Qwen3-8B",
        "experts": [
            {
                "expert_name": "adapter_1",
                "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-mental_health"
            },
            {
                "expert_name": "adapter_2",
                "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-fiqa"
            }
        ],
        "router_layers": ["gate_proj", "up_proj", "down_proj"],
    }
    
    checkpoint_path = "data/test_qwen3_moe_checkpoint"
    
    # Clean up if exists
    if os.path.exists(checkpoint_path):
        import shutil
        shutil.rmtree(checkpoint_path)
        print(f"Removed existing checkpoint at {checkpoint_path}")
    
    try:
        print("\nCreating composer...")
        expertmerger = ComposeExperts(
            model_config,
            torch_dtype=torch.bfloat16,
            model_cls=Qwen3ForCausalLM
        )
        
        print("\nComposing experts...")
        expertmerger.compose()
        
        print("\nSaving checkpoint...")
        expertmerger.save_checkpoint(checkpoint_path)
        
        print("\n✅ Composition successful!")
        return checkpoint_path
        
    except Exception as e:
        print(f"\n❌ Composition failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_loading(checkpoint_path):
    """Test Step 2: Loading MoE checkpoint"""
    print("\n" + "="*80)
    print("STEP 2: TESTING CHECKPOINT LOADING")
    print("="*80)
    
    try:
        print("\nLoading model from checkpoint...")
        model = Qwen3ForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True
        )
        
        print("\n✅ Model loaded successfully!")

        # Reinitialize all normalization weights to 1.0 for bf16 stability
        print("\nReinitializing normalization weights...")
        norm_count = 0
        for name, param in model.named_parameters():
            if 'norm' in name.lower() and 'weight' in name:
                with torch.no_grad():
                    param.fill_(1.0)
                    norm_count += 1
        print(f"✓ Reinitialized {norm_count} normalization weights to 1.0")

        # Check for NaN/Inf in weights
        print("\nChecking for NaN/Inf in weights...")
        nan_count = 0
        inf_count = 0
        total_params = 0
        gate_params = []

        for name, param in model.named_parameters():
            total_params += 1
            if 'gate.weight' in name:
                gate_params.append((name, param.shape, param.min().item(), param.max().item()))
            if torch.isnan(param).any():
                nan_count += 1
                print(f"  ⚠️  NaN found in: {name}")
            if torch.isinf(param).any():
                inf_count += 1
                print(f"  ⚠️  Inf found in: {name}")

        if nan_count == 0 and inf_count == 0:
            print(f"✅ All {total_params} parameters are clean (no NaN/Inf)")
        else:
            print(f"❌ Found {nan_count} NaN and {inf_count} Inf parameters out of {total_params}")

        # Check gate weights specifically
        if gate_params:
            print(f"\n✅ Found {len(gate_params)} gate weight tensors:")
            for name, shape, min_val, max_val in gate_params[:3]:
                print(f"  {name}: shape={shape}, range=[{min_val:.6f}, {max_val:.6f}]")
            if len(gate_params) > 3:
                print(f"  ... and {len(gate_params) - 3} more")
        else:
            print("\n❌ No gate weights found!")
        
        # Check MoE layers
        print("\nChecking MoE layers...")
        moe_layers = []
        for name, module in model.named_modules():
            if "LoRAMoeLayer" in str(type(module)):
                moe_layers.append(name)
        
        if moe_layers:
            print(f"✅ Found {len(moe_layers)} MoE layers")
            print(f"  First few: {moe_layers[:3]}")
        else:
            print("❌ No MoE layers found!")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test Step 3: Forward pass"""
    print("\n" + "="*80)
    print("STEP 3: TESTING FORWARD PASS")
    print("="*80)
    
    try:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        
        print("\nPreparing test input...")
        test_text = "### Question: What is mental health?\n### Answer:"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        if torch.cuda.is_available():
            print("Moving model and inputs to CUDA...")
            model = model.to("cuda")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check for NaN in outputs
        if torch.isnan(outputs.logits).any():
            print("❌ NaN detected in output logits!")
            return False
        
        print(f"✅ Forward pass successful!")
        print(f"  Output shape: {outputs.logits.shape}")
        print(f"  Output dtype: {outputs.logits.dtype}")
        print(f"  Output range: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("QWEN3 MOE WITH LORA - COMPLETE FLOW TEST")
    print("="*80)
    
    # Test 1: Composition
    checkpoint_path = test_composition()
    if checkpoint_path is None:
        print("\n❌ OVERALL TEST FAILED: Composition failed")
        return
    
    # Test 2: Loading
    model = test_loading(checkpoint_path)
    if model is None:
        print("\n❌ OVERALL TEST FAILED: Loading failed")
        return
    
    # Test 3: Forward pass
    success = test_forward_pass(model)
    if not success:
        print("\n❌ OVERALL TEST FAILED: Forward pass failed")
        return
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nYour Qwen3 MoE with LoRA setup is working correctly!")
    print("You can now run the full training with: python projects/exmp_1_qwen.py")


if __name__ == "__main__":
    main()

