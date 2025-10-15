"""
Test script for the alpha MoE training implementation.
This script validates that the alpha coefficients are properly initialized and trainable.
"""

import sys
import os
import torch
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_alpha_moe import AlphaMoEModel, AlphaMoELinear, load_safe_samples, create_mixed_dataset
from mergoo.models.modeling_mistral import MistralForCausalLM


def test_alpha_layer_creation():
    """Test if alpha layers are properly created"""
    print("=== Testing Alpha Layer Creation ===")
    
    # Try to find an existing MoE model
    moe_paths = [
        "projects/data/mistral_lora_moe_all_XL_22apater_top2",
        "data/mistral_lora_moe_all_XL_22apater_top2",
        "projects/data/mistral_lora_moe_test",
        "data/mistral_lora_moe_test"
    ]
    
    model_path = None
    for path in moe_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("No MoE model found. Available models:")
        if os.path.exists("projects/data"):
            for item in os.listdir("projects/data"):
                if "moe" in item.lower():
                    print(f"  - projects/data/{item}")
        return False
    
    print(f"Using model: {model_path}")
    
    try:
        # Load model
        model = MistralForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # Use CPU for testing
            torch_dtype=torch.float32,
        )
        
        # Create alpha model
        alpha_model = AlphaMoEModel(model)
        
        # Check if alpha layers were created
        if alpha_model.alpha_layers:
            print(f"✓ Successfully created {len(alpha_model.alpha_layers)} alpha layers")
            for layer_name in alpha_model.alpha_layers.keys():
                print(f"  - {layer_name}")
        else:
            print("✗ No alpha layers were created")
            return False
        
        # Check trainable parameters
        total_params = sum(p.numel() for p in alpha_model.parameters())
        trainable_params = sum(p.numel() for p in alpha_model.parameters() if p.requires_grad)
        
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Trainable percentage: {100 * trainable_params / total_params:.4f}%")
        
        if trainable_params > 0:
            print("✓ Alpha parameters are trainable")
        else:
            print("✗ No trainable parameters found")
            return False
        
        # Test forward pass
        print("Testing forward pass...")
        dummy_input = torch.randint(0, 1000, (1, 10))  # batch_size=1, seq_len=10
        
        with torch.no_grad():
            output = alpha_model(dummy_input)
            print(f"✓ Forward pass successful, output shape: {output.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False


def test_data_loading():
    """Test data loading functionality"""
    print("\n=== Testing Data Loading ===")
    
    safe_data_path = "projects/evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json"
    
    if not os.path.exists(safe_data_path):
        print(f"✗ Safe data file not found: {safe_data_path}")
        print("Available evaluation files:")
        if os.path.exists("projects/evals_output"):
            for item in os.listdir("projects/evals_output"):
                if item.endswith(".json") and "gpt4o" in item:
                    print(f"  - {item}")
        return False
    
    try:
        # Test safe samples loading
        safe_samples = load_safe_samples(safe_data_path)
        print(f"✓ Loaded {len(safe_samples)} safe samples")
        
        if safe_samples:
            sample = safe_samples[0]
            print(f"✓ Sample structure: {list(sample.keys())}")
        
        # Test dataset creation (without Alpaca to avoid network dependency)
        print("Testing dataset creation...")
        dataset = create_mixed_dataset(safe_data_path, alpaca_percentage=0.0)  # No Alpaca for testing
        
        print(f"✓ Created dataset with {len(dataset['train'])} train and {len(dataset['val'])} val samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during data loading: {e}")
        return False


def main():
    """Run all tests"""
    print("Alpha MoE Training - Test Suite")
    print("=" * 40)
    
    # Test 1: Alpha layer creation
    test1_passed = test_alpha_layer_creation()
    
    # Test 2: Data loading
    test2_passed = test_data_loading()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Alpha layer creation: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Data loading: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The implementation is ready to use.")
        print("\nTo run training:")
        print("  ./projects/run_alpha_training.sh [moe_model_path]")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    main()
