"""
Generate text using trained alpha MoE model.

This script loads a trained alpha MoE model and generates text responses
to given prompts, allowing you to test the effect of the trained alpha coefficients.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mergoo.models.modeling_mistral import MistralForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import gc


class AlphaMoELinear(nn.Module):
    """
    Enhanced MoE Linear layer with trainable alpha coefficients.
    V2: Uses single alpha vector over the entire mixture
    """
    def __init__(self, original_moe_layer, alpha_values=None):
        super().__init__()
        self.original_layer = original_moe_layer

        # Copy all attributes from original layer
        for attr_name in dir(original_moe_layer):
            if not attr_name.startswith('_') and not callable(getattr(original_moe_layer, attr_name)):
                try:
                    setattr(self, attr_name, getattr(original_moe_layer, attr_name))
                except:
                    pass

        # Store base layer weights (frozen)
        if hasattr(original_moe_layer, 'base_layer'):
            self.register_buffer('base_weight', original_moe_layer.base_layer.weight.data.clone())

        # Initialize alpha vector
        if hasattr(original_moe_layer, 'base_layer'):
            output_dim = original_moe_layer.base_layer.out_features
        else:
            # Fallback: try to infer from first LoRA layer
            first_adapter = list(original_moe_layer.active_adapters)[0]
            if first_adapter in original_moe_layer.lora_B:
                output_dim = original_moe_layer.lora_B[first_adapter].out_features
            else:
                output_dim = 4096  # Default fallback
        
        # Initialize alpha coefficient
        if alpha_values is not None:
            # Load from saved values
            self.alpha_coefficient = nn.Parameter(torch.tensor(alpha_values, dtype=torch.float32))
        else:
            # Default initialization
            self.alpha_coefficient = nn.Parameter(torch.full((output_dim,), 0.5, dtype=torch.float32))

    def forward(self, x, *args, **kwargs):
        """Modified forward pass with alpha coefficients"""
        previous_dtype = x.dtype
        gate_logits = self.original_layer.gate(x)  # b,s,N
        gate_logits = F.softsign(gate_logits).to(previous_dtype)
        weights, selected_experts = torch.topk(
            gate_logits, self.original_layer.num_experts_per_tok
        )

        # Base layer output
        base_result = self.original_layer.base_layer(x, *args, **kwargs)
        
        # Initialize mixture result with zeros
        mixture_result = torch.zeros_like(base_result)

        # Apply LoRA adapters to compute mixture output
        for ix, active_adapter in enumerate(self.original_layer.active_adapters):
            if active_adapter not in self.original_layer.lora_A.keys():
                continue

            lora_A = self.original_layer.lora_A[active_adapter]
            lora_B = self.original_layer.lora_B[active_adapter]
            dropout = self.original_layer.lora_dropout[active_adapter]
            scaling = self.original_layer.scaling[active_adapter]

            x_typed = x.to(lora_A.weight.dtype)
            batch_idx, tok_idx, expert_idx = torch.where(selected_experts == ix)
            
            if len(batch_idx) == 0:  # No tokens assigned to this expert
                continue
                
            x_adapter = x_typed[batch_idx, tok_idx]

            # Compute LoRA delta
            delta_moe = lora_B(lora_A(dropout(x_adapter))) * scaling

            # Ensure weights are on the same device as delta_moe
            expert_weights = weights[batch_idx, tok_idx, expert_idx].unsqueeze(-1)
            expert_weights = expert_weights.to(delta_moe.device)

            # Add weighted delta to mixture result
            mixture_result[batch_idx, tok_idx] += delta_moe * expert_weights

        # Apply single alpha vector weighting
        alpha = self.alpha_coefficient.to(base_result.device)
        
        # Element-wise multiplication with alpha vector
        result = alpha.unsqueeze(0).unsqueeze(0) * base_result + \
                (1 - alpha).unsqueeze(0).unsqueeze(0) * (base_result + mixture_result)

        result = result.to(previous_dtype)
        return result


class AlphaMoEModel(nn.Module):
    """Model that replaces MoE layers with alpha-coefficient enhanced versions"""
    def __init__(self, base_moe_model, alpha_coefficients=None):
        super().__init__()
        self.base_model = base_moe_model
        self.alpha_layers = {}
        self.alpha_coefficients = alpha_coefficients or {}

        # Replace MoE layers with alpha-enhanced versions
        self._replace_moe_layers()

    def _replace_moe_layers(self):
        """Replace MoE layers with alpha-enhanced versions"""
        def replace_layer(module, name_parts):
            for name, child in module.named_children():
                full_name = '.'.join(name_parts + [name])

                # Check if this is a MoE layer
                if (hasattr(child, 'active_adapters') and
                    hasattr(child, 'lora_A') and
                    hasattr(child, 'base_layer')):

                    print(f"Replacing MoE layer: {full_name}")
                    # Get alpha values for this layer if available
                    alpha_values = self.alpha_coefficients.get(full_name, None)
                    alpha_layer = AlphaMoELinear(child, alpha_values)
                    setattr(module, name, alpha_layer)
                    self.alpha_layers[full_name] = alpha_layer
                else:
                    # Recursively check children
                    replace_layer(child, name_parts + [name])

        replace_layer(self.base_model, [])

    def forward(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.base_model(*args, **kwargs)

    def print_alpha_values(self, title="Alpha values"):
        """Print current alpha values"""
        print(f"\n{title}:")
        for layer_name, alpha_layer in self.alpha_layers.items():
            alpha_vec = alpha_layer.alpha_coefficient
            print(f"  {layer_name}: vector shape={alpha_vec.shape}, mean={alpha_vec.mean().item():.4f}, std={alpha_vec.std().item():.4f}")


def load_alpha_model(model_path: str, alpha_coefficients_path: str = None, device: str = "cuda"):
    """Load the trained alpha MoE model"""
    logger.info(f"Loading base MoE model from {model_path}")
    
    # Load base model
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None
    )
    
    # Load alpha coefficients if provided
    alpha_coefficients = {}
    if alpha_coefficients_path and os.path.exists(alpha_coefficients_path):
        logger.info(f"Loading alpha coefficients from {alpha_coefficients_path}")
        with open(alpha_coefficients_path, 'r') as f:
            alpha_coefficients = json.load(f)
    else:
        logger.warning("No alpha coefficients file provided or found. Using default values.")
    
    # Create alpha model
    alpha_model = AlphaMoEModel(model, alpha_coefficients)
    
    if device != "cuda":
        alpha_model = alpha_model.to(device)
    
    alpha_model.eval()
    return alpha_model


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, do_sample: bool = True):
    """Generate text using the alpha MoE model"""
    # Format prompt
    formatted_prompt = f"### Question: {prompt}\n### Answer:"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response (remove input prompt)
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained alpha MoE model")
    parser.add_argument("--model_path", type=str, 
                       default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/alpha_moe_trained",
                       help="Path to the trained alpha MoE model")
    parser.add_argument("--alpha_coefficients", type=str,
                       help="Path to alpha coefficients JSON file (optional)")
    parser.add_argument("--prompt", type=str,
                       default="What is the capital of France?",
                       help="Prompt to generate text for")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Auto-detect alpha coefficients file if not provided
    if not args.alpha_coefficients:
        alpha_file = os.path.join(args.model_path, "alpha_coefficients.json")
        if os.path.exists(alpha_file):
            args.alpha_coefficients = alpha_file
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load alpha model
    alpha_model = load_alpha_model(args.model_path, args.alpha_coefficients, device)
    
    # Print alpha values
    alpha_model.print_alpha_values("Loaded Alpha Values")
    
    if args.interactive:
        # Interactive mode
        print("\n=== Interactive Alpha MoE Text Generation ===")
        print("Type 'quit' to exit")
        
        while True:
            prompt = input("\nEnter your prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print(f"\nGenerating response...")
            response = generate_text(alpha_model, tokenizer, prompt, 
                                   args.max_new_tokens, args.temperature)
            print(f"\nResponse: {response}")
    else:
        # Single generation
        print(f"\nPrompt: {args.prompt}")
        print(f"Generating response...")
        response = generate_text(alpha_model, tokenizer, args.prompt, 
                               args.max_new_tokens, args.temperature)
        print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
