#!/usr/bin/env python3
"""
Example script showing how to use the trained alpha MoE model for text generation.

This script demonstrates:
1. Loading a trained alpha MoE model
2. Generating text with different prompts
3. Comparing responses with different alpha values (if available)
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from generate_with_alpha_moe import load_alpha_model, generate_text
from transformers import AutoTokenizer
import torch
import json

def main():
    # Configuration
    model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/alpha_moe_trained"
    alpha_coefficients_path = os.path.join(model_path, "alpha_coefficients.json")
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        print("Please train the alpha MoE model first using train_alpha_moe_v2.py")
        return
    
    print("=== Alpha MoE Text Generation Example ===\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading trained alpha MoE model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha_model = load_alpha_model(model_path, alpha_coefficients_path, device)
    
    # Print alpha statistics
    alpha_model.print_alpha_values("Trained Alpha Coefficients")
    
    # Example prompts to test
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "How do you make a paper airplane?",
        "What are the benefits of renewable energy?",
    ]
    
    print(f"\n=== Generating Responses ===\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"--- Example {i} ---")
        print(f"Prompt: {prompt}")
        
        # Generate response
        response = generate_text(
            alpha_model, 
            tokenizer, 
            prompt, 
            max_new_tokens=150, 
            temperature=0.7
        )
        
        print(f"Response: {response}")
        print()
    
    # Show alpha coefficient statistics
    if os.path.exists(alpha_coefficients_path):
        print("=== Alpha Coefficient Analysis ===")
        with open(alpha_coefficients_path, 'r') as f:
            alpha_data = json.load(f)
        
        for layer_name, alpha_values in alpha_data.items():
            alpha_tensor = torch.tensor(alpha_values)
            print(f"Layer: {layer_name}")
            print(f"  Shape: {alpha_tensor.shape}")
            print(f"  Mean: {alpha_tensor.mean().item():.4f}")
            print(f"  Std: {alpha_tensor.std().item():.4f}")
            print(f"  Min: {alpha_tensor.min().item():.4f}")
            print(f"  Max: {alpha_tensor.max().item():.4f}")
            print(f"  Values close to 0.5: {((alpha_tensor > 0.4) & (alpha_tensor < 0.6)).sum().item()}/{len(alpha_values)}")
            print()
    
    print("=== Generation Complete ===")
    print("\nTo run interactively, use:")
    print("python projects/generate_with_alpha_moe.py --interactive")


if __name__ == "__main__":
    main()
