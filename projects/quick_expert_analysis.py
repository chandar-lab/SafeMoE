#!/usr/bin/env python3
"""
Quick Expert Analysis Tool
A simple script to analyze expert activation patterns in your MoE model.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from mergoo.models.modeling_mistral import MistralForCausalLM
from collections import Counter
from scipy.stats import entropy
import json
import matplotlib.pyplot as plt
import random
import glob

def analyze_expert_activations(model_path, prompts, layers=None, max_tokens=50):
    """
    Analyze expert activation patterns for given prompts
    
    Args:
        model_path: Path to your MoE model
        prompts: List of prompts to analyze
        layers: List of layer indices to analyze (default: [0, 8, 16, 24, 31])
        max_tokens: Maximum tokens to generate per prompt
    
    Returns:
        Dictionary with analysis results
    """
    
    if layers is None:
        layers = [0, 8, 16, 24, 31]
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    print(f"Model info: {model.config.num_experts} experts, {model.config.num_hidden_layers} layers")
    
    # Setup hooks
    expert_data = {}
    hooks = []
    
    def register_hook(layer_idx):
        gate_name = f'model.layers.{layer_idx}.mlp.gate_proj.gate'
        for name, module in model.named_modules():
            if name == gate_name:
                expert_data[layer_idx] = []
                
                def hook_fn(_, __, output):
                    weights = F.softmax(output, dim=-1)
                    _, top_experts = torch.topk(weights, k=1, dim=-1)
                    selections = top_experts.squeeze(-1).detach().cpu().flatten().tolist()
                    expert_data[layer_idx].extend(selections)
                
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                print(f'  Registered hook for layer {layer_idx}')
                break
    
    # Register hooks
    for layer_idx in layers:
        register_hook(layer_idx)
    
    # Process prompts
    print(f"\nProcessing {len(prompts)} prompts...")
    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i+1}: {prompt[:60]}...")
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # Analyze results
    print(f"\n{'='*60}")
    print("EXPERT ACTIVATION ANALYSIS")
    print('='*60)
    
    results = {}
    for layer_idx in layers:
        if layer_idx in expert_data and expert_data[layer_idx]:
            selections = expert_data[layer_idx]
            counts = Counter(selections)
            total = len(selections)
            
            print(f"\nLayer {layer_idx}:")
            print(f"  Total tokens: {total}")
            
            # Calculate distribution
            distribution = {}
            for expert_id in range(model.config.num_experts):
                count = counts.get(expert_id, 0)
                prob = count / total if total > 0 else 0
                distribution[expert_id] = prob
                print(f"  Expert {expert_id}: {count:3d} tokens ({prob:.1%})")
            
            # Calculate entropy
            probs = [distribution[i] for i in range(model.config.num_experts)]
            probs_filtered = [p for p in probs if p > 0]
            
            if len(probs_filtered) > 1:
                sel_entropy = entropy(probs_filtered, base=2)
                max_entropy = np.log2(model.config.num_experts)
                entropy_ratio = sel_entropy / max_entropy
            else:
                sel_entropy = 0.0
                entropy_ratio = 0.0
            
            print(f"  Entropy: {sel_entropy:.3f} bits (ratio: {entropy_ratio:.3f})")
            
            # Interpret balance
            if entropy_ratio > 0.9:
                balance = "Very well balanced"
            elif entropy_ratio > 0.7:
                balance = "Moderately balanced"
            elif entropy_ratio > 0.5:
                balance = "Some expert preference"
            else:
                balance = "Unbalanced"
            
            print(f"  Status: {balance}")
            
            results[f'layer_{layer_idx}'] = {
                'distribution': distribution,
                'entropy': sel_entropy,
                'entropy_ratio': entropy_ratio,
                'total_tokens': total,
                'balance_status': balance
            }
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        
        entropies = [results[k]['entropy'] for k in results.keys()]
        ratios = [results[k]['entropy_ratio'] for k in results.keys()]
        
        print(f"Average entropy: {np.mean(entropies):.3f} bits")
        print(f"Average entropy ratio: {np.mean(ratios):.3f}")
        
        well_balanced = sum(1 for r in ratios if r > 0.9)
        moderately_balanced = sum(1 for r in ratios if 0.7 < r <= 0.9)
        unbalanced = sum(1 for r in ratios if r <= 0.7)
        
        print(f"Balance summary:")
        print(f"  Well balanced: {well_balanced} layers")
        print(f"  Moderately balanced: {moderately_balanced} layers")
        print(f"  Unbalanced: {unbalanced} layers")
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    
    return results

def analyze_prompts_with_model(model, tokenizer, prompts, layers=None, max_tokens=50):
    """
    Analyze expert activation patterns for given prompts using pre-loaded model

    Args:
        model: Pre-loaded MoE model
        tokenizer: Pre-loaded tokenizer
        prompts: List of prompts to analyze
        layers: List of layer indices to analyze (default: [0, 8, 16, 24, 31])
        max_tokens: Maximum tokens to generate per prompt

    Returns:
        Dictionary with analysis results
    """

    if layers is None:
        layers = [0, 8, 16, 24, 31]

    # Setup hooks
    expert_data = {}
    hooks = []

    def register_hook(layer_idx):
        gate_name = f'model.layers.{layer_idx}.mlp.gate_proj.gate'
        for name, module in model.named_modules():
            if name == gate_name:
                expert_data[layer_idx] = []

                def hook_fn(_, __, output):
                    weights = F.softmax(output, dim=-1)
                    _, top_experts = torch.topk(weights, k=1, dim=-1)
                    selections = top_experts.squeeze(-1).detach().cpu().flatten().tolist()
                    expert_data[layer_idx].extend(selections)

                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                break

    # Register hooks
    for layer_idx in layers:
        register_hook(layer_idx)

    # Process prompts
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

    # Analyze results
    results = {}
    for layer_idx in layers:
        if layer_idx in expert_data and expert_data[layer_idx]:
            selections = expert_data[layer_idx]
            counts = Counter(selections)
            total = len(selections)

            # Calculate distribution
            distribution = {}
            for expert_id in range(model.config.num_experts):
                count = counts.get(expert_id, 0)
                prob = count / total if total > 0 else 0
                distribution[expert_id] = prob

            # Calculate entropy
            probs = [distribution[i] for i in range(model.config.num_experts)]
            probs_filtered = [p for p in probs if p > 0]

            if len(probs_filtered) > 1:
                sel_entropy = entropy(probs_filtered, base=2)
                max_entropy = np.log2(model.config.num_experts)
                entropy_ratio = sel_entropy / max_entropy
            else:
                sel_entropy = 0.0
                entropy_ratio = 0.0

            # Interpret balance
            if entropy_ratio > 0.9:
                balance = "Very well balanced"
            elif entropy_ratio > 0.7:
                balance = "Moderately balanced"
            elif entropy_ratio > 0.5:
                balance = "Some expert preference"
            else:
                balance = "Unbalanced"

            results[f'layer_{layer_idx}'] = {
                'distribution': distribution,
                'entropy': sel_entropy,
                'entropy_ratio': entropy_ratio,
                'total_tokens': total,
                'balance_status': balance
            }

    # Cleanup
    for hook in hooks:
        hook.remove()

    return results

def analyze_multiple_test_files(model_path, data_dir, layers=None, samples_per_file=10, max_tokens=50):
    """
    Analyze expert activation patterns for multiple test files

    Args:
        model_path: Path to your MoE model
        data_dir: Directory containing test JSON files
        layers: List of layer indices to analyze (default: [0, 8, 16, 24, 31])
        samples_per_file: Number of samples to pick from each file
        max_tokens: Maximum tokens to generate per prompt

    Returns:
        Dictionary with analysis results for each file
    """

    if layers is None:
        layers = [0, 8, 16, 24, 31]

    # Find all test files
    test_files = glob.glob(os.path.join(data_dir, "test_*.json"))
    test_files.sort()

    print(f"Found {len(test_files)} test files")
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print(f"Model info: {model.config.num_experts} experts, {model.config.num_hidden_layers} layers")

    all_results = {}

    for test_file in test_files:
        file_name = os.path.basename(test_file).replace('.json', '')
        print(f"\nProcessing {file_name}...")

        # Load and sample data
        with open(test_file, 'r') as f:
            data = json.load(f)

        # Randomly sample prompts (set seed for reproducibility)
        random.seed(42)
        if len(data) > samples_per_file:
            sampled_data = random.sample(data, samples_per_file)
        else:
            sampled_data = data

        prompts = [item['prompt'] for item in sampled_data]
        print(f"  Selected {len(prompts)} prompts from {len(data)} total")

        # Analyze this file's prompts (reuse loaded model)
        results = analyze_prompts_with_model(model, tokenizer, prompts, layers, max_tokens)
        all_results[file_name] = results

    return all_results

def create_expert_distribution_plots(all_results, output_dir, num_experts=22):
    """
    Create bar plots for expert distributions for each test file

    Args:
        all_results: Results from analyze_multiple_test_files
        output_dir: Directory to save plots
        num_experts: Number of experts in the model
    """

    os.makedirs(output_dir, exist_ok=True)

    # Create individual plots for each file
    for file_name, file_results in all_results.items():
        if not file_results:
            continue

        # Calculate average distribution across all layers for this file
        avg_distribution = np.zeros(num_experts)
        layer_count = 0

        for _, layer_data in file_results.items():
            if 'distribution' in layer_data:
                for expert_id in range(num_experts):
                    avg_distribution[expert_id] += layer_data['distribution'].get(expert_id, 0)
                layer_count += 1

        if layer_count > 0:
            avg_distribution /= layer_count

        # Create bar plot
        plt.figure(figsize=(12, 6))
        expert_ids = list(range(num_experts))
        bars = plt.bar(expert_ids, avg_distribution, alpha=0.7, color='skyblue', edgecolor='navy')

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.title(f'Average Expert Distribution - {file_name.replace("test_", "").replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Expert ID', fontsize=12)
        plt.ylabel('Average Activation Probability', fontsize=12)
        plt.xticks(expert_ids)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, f'{file_name}_expert_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot: {plot_path}")

    # Create summary plot with all files
    plt.figure(figsize=(16, 10))

    file_names = []
    all_distributions = []

    for file_name, file_results in all_results.items():
        if not file_results:
            continue

        # Calculate average distribution across all layers for this file
        avg_distribution = np.zeros(num_experts)
        layer_count = 0

        for _, layer_data in file_results.items():
            if 'distribution' in layer_data:
                for expert_id in range(num_experts):
                    avg_distribution[expert_id] += layer_data['distribution'].get(expert_id, 0)
                layer_count += 1

        if layer_count > 0:
            avg_distribution /= layer_count
            file_names.append(file_name.replace("test_", "").replace("_", " ").title())
            all_distributions.append(avg_distribution)

    if all_distributions:
        # Create heatmap-style visualization
        distributions_matrix = np.array(all_distributions)

        im = plt.imshow(distributions_matrix, cmap='Blues', aspect='auto')
        plt.colorbar(im, label='Average Activation Probability')

        plt.title('Expert Distribution Heatmap Across All Test Files', fontsize=16, fontweight='bold')
        plt.xlabel('Expert ID', fontsize=12)
        plt.ylabel('Test Files', fontsize=12)

        plt.xticks(range(num_experts))
        plt.yticks(range(len(file_names)), file_names, fontsize=10)

        # Add text annotations
        for i in range(len(file_names)):
            for j in range(num_experts):
                plt.text(j, i, f'{distributions_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()

        # Save summary plot
        summary_plot_path = os.path.join(output_dir, 'all_files_expert_distribution_heatmap.png')
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved summary heatmap: {summary_plot_path}")

    return all_distributions

def main():
    """Main function with predefined analysis scenarios"""
    
    model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250"
    
    print("Expert Activation Analysis Tool")
    print("="*50)
    print("Choose an analysis scenario:")
    print("1. Drug abuse domain analysis (default)")
    print("2. General knowledge questions")
    print("3. Custom prompt")
    print("4. Comprehensive analysis (multiple domains)")
    print("5. Multi-file analysis (PKU SaferLHF dataset)")

    choice = input("\nEnter choice (1-5, default=1): ").strip() or "1"
    
    if choice == "1":
        # Drug abuse domain
        prompts = [
            "### Question: What are the long-term effects of cocaine addiction?\n### Answer:",
            "### Question: How does alcohol abuse affect the brain?\n### Answer:",
            "### Question: What are effective treatments for opioid addiction?\n### Answer:",
            "### Question: How can we prevent drug abuse in teenagers?\n### Answer:",
            "### Question: What are the signs of drug overdose?\n### Answer:"
        ]
        output_dir = "drug_abuse_analysis"
        
    elif choice == "2":
        # General knowledge
        prompts = [
            "### Question: What is the capital of France?\n### Answer:",
            "### Question: How does photosynthesis work?\n### Answer:",
            "### Question: What is machine learning?\n### Answer:",
            "### Question: Explain the theory of relativity.\n### Answer:",
            "### Question: What causes climate change?\n### Answer:"
        ]
        output_dir = "general_knowledge_analysis"
        
    elif choice == "3":
        # Custom prompt
        custom_prompt = input("Enter your prompt: ").strip()
        if not custom_prompt:
            custom_prompt = "### Question: What is artificial intelligence?\n### Answer:"
        prompts = [custom_prompt]
        output_dir = "custom_analysis"
        
    elif choice == "4":
        # Comprehensive analysis
        prompts = [
            # Drug abuse
            "### Question: What are the effects of drug abuse?\n### Answer:",
            "### Question: How to treat addiction?\n### Answer:",
            # Medical
            "### Question: What is diabetes?\n### Answer:",
            "### Question: How does the heart work?\n### Answer:",
            # General
            "### Question: What is artificial intelligence?\n### Answer:",
            "### Question: Explain quantum physics.\n### Answer:",
            # Technical
            "### Question: How do computers work?\n### Answer:",
            "### Question: What is programming?\n### Answer:"
        ]
        output_dir = "comprehensive_analysis"

    elif choice == "5":
        # Multi-file analysis
        data_dir = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed"
        samples_per_file = int(input("Number of samples per file (default=10): ").strip() or "10")

        print(f"\nRunning multi-file analysis on {data_dir}")
        print(f"Samples per file: {samples_per_file}")

        # Run multi-file analysis
        all_results = analyze_multiple_test_files(model_path, data_dir, samples_per_file=samples_per_file, layers=12)

        # Create output directory
        output_dir = "multi_file_analysis_2"
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        with open(f"{output_dir}/detailed_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

        # Create visualizations
        print("\nCreating visualizations...")
        distributions = create_expert_distribution_plots(all_results, output_dir)

        # Save distribution arrays
        np.save(f"{output_dir}/expert_distributions.npy", np.array(distributions))

        # Create summary report
        summary = f"""# Multi-File Expert Analysis Results

## Dataset: PKU SaferLHF Processed
## Samples per file: {samples_per_file}
## Files analyzed: {len(all_results)}

## Files processed:
"""

        for file_name, file_results in all_results.items():
            if file_results:
                entropies = [file_results[k]['entropy'] for k in file_results.keys()]
                ratios = [file_results[k]['entropy_ratio'] for k in file_results.keys()]

                summary += f"""
### {file_name.replace('test_', '').replace('_', ' ').title()}
- Average entropy: {np.mean(entropies):.3f} bits
- Average entropy ratio: {np.mean(ratios):.3f}
- Layers analyzed: {len(file_results)}
"""

        with open(f"{output_dir}/multi_file_summary.md", 'w') as f:
            f.write(summary)

        print(f"\nMulti-file analysis complete!")
        print(f"Results saved to {output_dir}/")
        print(f"  - detailed_results.json: Complete numerical results")
        print(f"  - expert_distributions.npy: Distribution arrays")
        print(f"  - multi_file_summary.md: Summary report")
        print(f"  - Individual plots for each file")
        print(f"  - all_files_expert_distribution_heatmap.png: Summary heatmap")

        return all_results

    else:
        print("Invalid choice, using drug abuse analysis...")
        choice = "1"
        prompts = [
            "### Question: What are the effects of drug abuse?\n### Answer:",
            "### Question: How to treat addiction?\n### Answer:"
        ]
        output_dir = "default_analysis"
    
    # Run analysis
    results = analyze_expert_activations(model_path, prompts)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    summary = f"""# Expert Analysis Results

## Analysis Type: {['', 'Drug Abuse Domain', 'General Knowledge', 'Custom Prompt', 'Comprehensive'][int(choice)]}

## Prompts Analyzed: {len(prompts)}

## Results Summary:
"""
    
    if results:
        entropies = [results[k]['entropy'] for k in results.keys()]
        ratios = [results[k]['entropy_ratio'] for k in results.keys()]
        
        summary += f"""
- Average entropy: {np.mean(entropies):.3f} bits
- Average entropy ratio: {np.mean(ratios):.3f}
- Well balanced layers: {sum(1 for r in ratios if r > 0.9)}
- Total layers analyzed: {len(results)}

## Layer Details:
"""
        
        for layer_key, data in results.items():
            layer_num = layer_key.split('_')[1]
            summary += f"""
### Layer {layer_num}
- Entropy: {data['entropy']:.3f} bits
- Entropy ratio: {data['entropy_ratio']:.3f}
- Status: {data['balance_status']}
- Expert distribution: {[f"Expert {i}: {data['distribution'][i]:.1%}" for i in range(len(data['distribution']))]}
"""
    
    with open(f"{output_dir}/summary.md", 'w') as f:
        f.write(summary)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - results.json: Detailed numerical results")
    print(f"  - summary.md: Human-readable summary")
    
    return results

if __name__ == "__main__":
    main()
