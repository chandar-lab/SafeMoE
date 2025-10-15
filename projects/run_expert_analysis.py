#!/usr/bin/env python3
"""
Simple script to run expert activation analysis on your MoE model.
This script provides easy-to-use functions for analyzing expert distributions and entropy.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from projects.expert_activation_analyzer import ExpertActivationAnalyzer, plot_expert_analysis
from transformers import AutoTokenizer
from mergoo.models.modeling_mistral import MistralForCausalLM
import torch
import json

def quick_analysis(model_path, prompt, layers=None, max_tokens=50):
    """
    Quick analysis function for a single prompt
    
    Args:
        model_path: Path to your MoE model
        prompt: Text prompt to analyze
        layers: List of layer indices to analyze (default: [0, 8, 16, 24])
        max_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary with analysis results
    """
    if layers is None:
        layers = [0, 8, 16, 24]
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    analyzer = ExpertActivationAnalyzer(model, tokenizer, layers)
    
    try:
        print(f"Processing prompt: {prompt[:100]}...")
        analyzer.process_text(prompt, max_tokens)
        
        results = {}
        print("\n" + "="*60)
        print("EXPERT ACTIVATION ANALYSIS")
        print("="*60)
        
        for layer_idx in layers:
            print(f"\nLayer {layer_idx}:")
            print("-" * 30)
            
            # Get distribution
            distribution, counts = analyzer.calculate_expert_distribution(layer_idx)
            if distribution:
                print("Expert Distribution:")
                for expert_id, prob in distribution.items():
                    count = counts.get(expert_id, 0)
                    print(f"  Expert {expert_id}: {prob:.3f} ({count} tokens)")
                
                results[f'layer_{layer_idx}_distribution'] = distribution
            
            # Get entropy
            sel_entropy = analyzer.calculate_entropy(layer_idx, 'selection')
            weight_entropy = analyzer.calculate_entropy(layer_idx, 'weights')
            
            print(f"Selection Entropy: {sel_entropy:.3f} bits" if sel_entropy else "Selection Entropy: N/A")
            print(f"Weight Entropy: {weight_entropy:.3f} bits" if weight_entropy else "Weight Entropy: N/A")
            
            results[f'layer_{layer_idx}_selection_entropy'] = sel_entropy
            results[f'layer_{layer_idx}_weight_entropy'] = weight_entropy
            
            # Get routing stats
            stats = analyzer.get_routing_statistics(layer_idx)
            if stats and stats['load_balance'] is not None:
                print(f"Load Balance: {stats['load_balance']:.3f}")
                results[f'layer_{layer_idx}_load_balance'] = stats['load_balance']
        
        # Create visualization
        plot_expert_analysis(analyzer, layers, save_dir='./quick_analysis_results')
        
        return results
        
    finally:
        analyzer.cleanup()

def analyze_drug_abuse_responses():
    """
    Specific analysis for drug abuse related prompts
    """
    model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_top1/checkpoint-3910"
    
    prompts = [
        "### Question: What are the long-term effects of cocaine addiction?\n### Answer:",
        "### Question: How does alcohol abuse affect the brain?\n### Answer:",
        "### Question: What are effective treatments for opioid addiction?\n### Answer:",
        "### Question: How can we prevent drug abuse in teenagers?\n### Answer:",
        "### Question: What are the signs of drug overdose?\n### Answer:"
    ]
    
    print("Analyzing drug abuse related prompts...")
    
    # Load model once
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    layers = [0, 8, 16, 24, 31]  # Include last layer
    analyzer = ExpertActivationAnalyzer(model, tokenizer, layers)
    
    try:
        # Process all prompts
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {i+1}: {prompt.split('?')[0]}?")
            analyzer.process_text(prompt, max_tokens=100)
        
        # Analyze results
        print("\n" + "="*80)
        print("COMPREHENSIVE DRUG ABUSE DOMAIN ANALYSIS")
        print("="*80)
        
        all_results = {}
        
        for layer_idx in layers:
            print(f"\nLayer {layer_idx} Analysis:")
            print("-" * 50)
            
            distribution, counts = analyzer.calculate_expert_distribution(layer_idx)
            if distribution:
                print("Expert Usage Distribution:")
                total_tokens = sum(counts.values())
                for expert_id, prob in distribution.items():
                    count = counts.get(expert_id, 0)
                    print(f"  Expert {expert_id}: {prob:.3f} ({count}/{total_tokens} tokens)")
                
                # Check if one expert dominates
                max_prob = max(distribution.values())
                dominant_expert = max(distribution, key=distribution.get)
                if max_prob > 0.8:
                    print(f"  ⚠️  Expert {dominant_expert} dominates with {max_prob:.1%} usage!")
                elif max_prob < 0.6:
                    print(f"  ✓ Balanced expert usage (max: {max_prob:.1%})")
                else:
                    print(f"  → Expert {dominant_expert} is preferred ({max_prob:.1%} usage)")
            
            # Entropy analysis
            sel_entropy = analyzer.calculate_entropy(layer_idx, 'selection')
            weight_entropy = analyzer.calculate_entropy(layer_idx, 'weights')
            
            print(f"Selection Entropy: {sel_entropy:.3f} bits" if sel_entropy else "Selection Entropy: N/A")
            print(f"Weight Entropy: {weight_entropy:.3f} bits" if weight_entropy else "Weight Entropy: N/A")
            
            # Interpret entropy
            if sel_entropy is not None:
                max_entropy = 1.0  # log2(2) for 2 experts
                entropy_ratio = sel_entropy / max_entropy
                if entropy_ratio > 0.9:
                    print("  → Very balanced expert selection")
                elif entropy_ratio > 0.7:
                    print("  → Moderately balanced expert selection")
                else:
                    print("  → Unbalanced expert selection (one expert dominates)")
            
            all_results[f'layer_{layer_idx}'] = {
                'distribution': distribution,
                'selection_entropy': sel_entropy,
                'weight_entropy': weight_entropy,
                'entropy_ratio': sel_entropy / 1.0 if sel_entropy else None
            }
        
        # Create comprehensive visualization
        plot_expert_analysis(analyzer, layers, save_dir='./drug_abuse_analysis')
        
        # Save results
        os.makedirs('./drug_abuse_analysis', exist_ok=True)
        with open('./drug_abuse_analysis/analysis_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nResults saved to ./drug_abuse_analysis/")
        return all_results
        
    finally:
        analyzer.cleanup()

def compare_expert_entropy_across_layers():
    """
    Compare entropy across all layers to understand routing patterns
    """
    model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_top1/checkpoint-3910"
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Analyze all layers
    all_layers = list(range(model.config.num_hidden_layers))
    analyzer = ExpertActivationAnalyzer(model, tokenizer, all_layers)
    
    try:
        # Test prompt
        prompt = "### Question: What are the effects of drug abuse on mental health?\n### Answer:"
        analyzer.process_text(prompt, max_tokens=150)
        
        print("\n" + "="*80)
        print("ENTROPY ANALYSIS ACROSS ALL LAYERS")
        print("="*80)
        
        entropies = []
        distributions = []
        
        for layer_idx in all_layers:
            sel_entropy = analyzer.calculate_entropy(layer_idx, 'selection')
            distribution, _ = analyzer.calculate_expert_distribution(layer_idx)
            
            entropies.append(sel_entropy if sel_entropy else 0)
            distributions.append(distribution)
            
            if layer_idx % 4 == 0:  # Print every 4th layer
                print(f"Layer {layer_idx:2d}: Entropy = {sel_entropy:.3f} bits" if sel_entropy else f"Layer {layer_idx:2d}: No data")
        
        # Find patterns
        print(f"\nEntropy Statistics:")
        valid_entropies = [e for e in entropies if e > 0]
        if valid_entropies:
            print(f"  Mean entropy: {sum(valid_entropies)/len(valid_entropies):.3f} bits")
            print(f"  Max entropy: {max(valid_entropies):.3f} bits (Layer {entropies.index(max(valid_entropies))})")
            print(f"  Min entropy: {min(valid_entropies):.3f} bits (Layer {entropies.index(min(valid_entropies))})")
        
        return entropies, distributions
        
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    print("Expert Activation Analysis Tool")
    print("="*50)
    print("Choose an analysis option:")
    print("1. Quick analysis with custom prompt")
    print("2. Drug abuse domain analysis")
    print("3. Entropy comparison across all layers")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        prompt = input("Enter your prompt: ").strip()
        if not prompt:
            prompt = "### Question: What are the effects of drug abuse on mental health?\n### Answer:"
        
        results = quick_analysis(
            "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_top1/checkpoint-3910",
            prompt
        )
        
    elif choice == "2":
        results = analyze_drug_abuse_responses()
        
    elif choice == "3":
        entropies, distributions = compare_expert_entropy_across_layers()
        
    else:
        print("Invalid choice. Running drug abuse analysis by default...")
        results = analyze_drug_abuse_responses()
