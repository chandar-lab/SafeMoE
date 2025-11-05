"""
Qwen3-8B LoRA MoE Composition Example

This script demonstrates how to create a Mixture of LoRA Experts (MoE) model
using Qwen3-8B as the base model. It replaces specified layers (like MLP layers)
with MoE routing layers that dynamically select between multiple LoRA adapters.

Usage:
    python examples/compose_lora_qwen3.py

The script will:
1. Load the base Qwen3-8B model
2. Load multiple LoRA adapters (fine-tuned on different tasks)
3. Create MoE routing layers for specified components
4. Save the composed model with mixture of LoRA experts
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM

# Output directory for the composed model
model_id = "data/qwen3_lora_moe"

# Configuration for LoRA MoE composition
config = {
    # Model architecture type
    "model_type": "qwen3",
    
    # Number of experts to activate per token
    # For example, 2 means each token will be processed by 2 out of N experts
    "num_experts_per_tok": 2,
    
    # Base model - the pretrained Qwen3-8B model
    "base_model": "Qwen/Qwen3-8B",
    
    # List of LoRA expert adapters
    # Each expert should be a LoRA adapter fine-tuned on a specific task/domain
    # Replace these with your actual LoRA adapter paths
    "experts": [
        {
            "expert_name": "adapter_1",
            "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-mental_health"  # e.g., fine-tuned on math
        },
        {
            "expert_name": "adapter_2", 
            "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-mental_health"  # e.g., fine-tuned on code
        },
        {
            "expert_name": "adapter_3",
            "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-mental_health"  # e.g., fine-tuned on reasoning
        },
        {
            "expert_name": "adapter_4",
            "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-mental_health"  # e.g., fine-tuned on creative writing
        },
    ],
    
    # Which layers to convert to MoE
    # These are the layer names where LoRA was applied during fine-tuning
    # Common choices for Qwen3:
    # - MLP layers: ["gate_proj", "up_proj", "down_proj"]
    # - Attention layers: ["q_proj", "k_proj", "v_proj", "o_proj"]
    # - All: ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    "router_layers": ['up_proj', 'gate_proj', 'down_proj'],
    
    
    # Which transformer layer indices to apply MoE
    # For Qwen3-8B (28 layers), you can specify which layers get MoE treatment
    # Examples:
    # - All layers: list(range(28))
    # - First 5 layers: [0, 1, 2, 3, 4]
    # - Last 5 layers: [23, 24, 25, 26, 27]
    # - Middle layers: [10, 11, 12, 13, 14, 15, 16, 17]
    # - Every other layer: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    "router_layers_index": list(range(28)),  # Apply to all 28 layers
}

# Create the MoE composer
print("=" * 80)
print("Creating Qwen3-8B LoRA MoE Model")
print("=" * 80)
print(f"\nBase Model: {config['base_model']}")
print(f"Number of Experts: {len(config['experts'])}")
print(f"Experts per Token: {config['num_experts_per_tok']}")
print(f"Router Layers: {config['router_layers']}")
print(f"Number of Transformer Layers with MoE: {len(config['router_layers_index'])}")
print("\nExperts:")
for i, expert in enumerate(config['experts'], 1):
    print(f"  {i}. {expert['expert_name']}: {expert['model_id']}")

# Compose the experts
print("\n" + "=" * 80)
print("Composing Experts...")
print("=" * 80)

expertmerger = ComposeExperts(
    config, 
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
    model_cls=Qwen3ForCausalLM   # Use our custom Qwen3 model class
)

# This will:
# 1. Load the base Qwen3-8B model
# 2. Load all LoRA adapters
# 3. Create MoE routing layers for specified components
# 4. Merge the weights appropriately
expertmerger.compose()

# Save the composed model
print("\n" + "=" * 80)
print(f"Saving composed model to: {model_id}")
print("=" * 80)
expertmerger.save_checkpoint(model_id)

print("\n" + "=" * 80)
print("✓ Successfully created Qwen3-8B LoRA MoE model!")
print("=" * 80)
print(f"\nYou can now load the model using:")
print(f"  from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM")
print(f"  model = Qwen3ForCausalLM.from_pretrained('{model_id}')")
print("\nThe model will automatically route tokens to the most appropriate")
print("expert adapters based on the learned routing weights.")

