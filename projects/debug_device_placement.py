"""
Debug script to check device placement in alpha MoE training
"""

import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_alpha_moe import AlphaMoEModel
from mergoo.models.modeling_mistral import MistralForCausalLM


def check_device_placement(model_path):
    """Check device placement of model components"""
    print("=== Device Placement Debug ===")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = MistralForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Create alpha model
    print("Creating alpha model...")
    alpha_model = AlphaMoEModel(model)
    
    # Check device placement
    print("\n=== Base Model Device Placement ===")
    for name, param in model.named_parameters():
        if "gate" in name or "lora" in name:
            print(f"{name}: {param.device}")
    
    print("\n=== Alpha Parameters Device Placement ===")
    for layer_name, alpha_layer in alpha_model.alpha_layers.items():
        for adapter_name, alpha in alpha_layer.alpha_coefficients.items():
            print(f"{layer_name}.{adapter_name}: {alpha.device}")
    
    print("\n=== Model Device Map ===")
    if hasattr(model, 'hf_device_map'):
        for layer, device in model.hf_device_map.items():
            print(f"{layer}: {device}")
    
    # Test forward pass with dummy input
    print("\n=== Testing Forward Pass ===")
    try:
        dummy_input = torch.randint(0, 1000, (1, 10))  # batch_size=1, seq_len=10
        
        # Move input to first available device
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            dummy_input = dummy_input.to(f"cuda:{device}")
            print(f"Input device: {dummy_input.device}")
        
        with torch.no_grad():
            output = alpha_model(dummy_input)
            print(f"✓ Forward pass successful, output device: {output.logits.device}")
            print(f"Output shape: {output.logits.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Print more detailed error info
        import traceback
        traceback.print_exc()
    
    return alpha_model


def fix_device_placement(alpha_model):
    """Attempt to fix device placement issues"""
    print("\n=== Attempting Device Placement Fix ===")
    
    # Get the primary device (where most parameters are)
    device_counts = {}
    for param in alpha_model.base_model.parameters():
        device = str(param.device)
        device_counts[device] = device_counts.get(device, 0) + 1
    
    primary_device = max(device_counts, key=device_counts.get)
    print(f"Primary device: {primary_device}")
    
    # Move alpha parameters to primary device
    for layer_name, alpha_layer in alpha_model.alpha_layers.items():
        for adapter_name, alpha in alpha_layer.alpha_coefficients.items():
            if str(alpha.device) != primary_device:
                print(f"Moving {layer_name}.{adapter_name} from {alpha.device} to {primary_device}")
                alpha_layer.alpha_coefficients[adapter_name] = alpha_layer.alpha_coefficients[adapter_name].to(primary_device)


def main():
    """Main debug function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug device placement")
    parser.add_argument("--model_path", type=str, 
                       default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250",
                       help="Path to MoE model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return
    
    # Check current GPU setup
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Check device placement
    alpha_model = check_device_placement(args.model_path)
    
    # Try to fix device issues
    fix_device_placement(alpha_model)
    
    # Test again after fix
    print("\n=== Testing After Device Fix ===")
    try:
        dummy_input = torch.randint(0, 1000, (1, 10))
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            dummy_input = dummy_input.to(f"cuda:{device}")
        
        with torch.no_grad():
            output = alpha_model(dummy_input)
            print(f"✓ Forward pass successful after fix!")
            
    except Exception as e:
        print(f"✗ Forward pass still failing: {e}")


if __name__ == "__main__":
    main()
