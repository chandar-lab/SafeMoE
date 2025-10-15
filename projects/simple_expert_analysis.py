#!/usr/bin/env python3
"""
Simple expert activation analysis for your MoE model
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
import matplotlib.pyplot as plt
import json

class SimpleExpertAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.expert_data = {}
        self.hooks = []
        
    def register_hooks(self, layer_indices):
        """Register hooks to capture expert routing data"""
        self.expert_data = {layer_idx: {'weights': [], 'selections': []} for layer_idx in layer_indices}
        
        for layer_idx in layer_indices:
            # Find gate modules for this layer - we need to hook all three MLP components
            gate_modules = []

            # Look for gate modules in gate_proj, up_proj, and down_proj
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                gate_name = f"model.layers.{layer_idx}.mlp.{proj_name}.gate"
                for name, module in self.model.named_modules():
                    if name == gate_name:
                        gate_modules.append((name, module))
                        print(f"Found gate module for layer {layer_idx}: {name}")
                        break

            if not gate_modules:
                print(f"Warning: Could not find any gate modules for layer {layer_idx}")
                continue
                
            def make_hook(layer_idx, gate_name):
                def hook_fn(module, input, output):
                    # Handle different output formats
                    if isinstance(output, tuple):
                        gate_logits = output[0]  # Take first element if tuple
                    else:
                        gate_logits = output

                    # Ensure we have the right shape
                    if gate_logits.dim() == 2:
                        # [batch*seq, num_experts] -> [batch, seq, num_experts]
                        batch_size = input[0].shape[0]
                        seq_len = input[0].shape[1]
                        gate_logits = gate_logits.view(batch_size, seq_len, -1)

                    weights = F.softmax(gate_logits, dim=-1)  # Convert to probabilities

                    # Get top expert selection
                    _, top_experts = torch.topk(weights, k=1, dim=-1)

                    # Store data (convert to float32 to avoid BFloat16 issues)
                    # Only store once per layer (use gate_proj as primary)
                    if 'gate_proj' in gate_name:
                        self.expert_data[layer_idx]['weights'].append(weights.detach().cpu().float())
                        self.expert_data[layer_idx]['selections'].append(top_experts.squeeze(-1).detach().cpu())

                return hook_fn

            # Register hooks for all gate modules in this layer
            for gate_name, gate_module in gate_modules:
                hook = gate_module.register_forward_hook(make_hook(layer_idx, gate_name))
                self.hooks.append(hook)
    
    def analyze_text(self, text, max_new_tokens=100):
        """Analyze expert usage for given text"""
        # Clear previous data
        for layer_idx in self.expert_data:
            self.expert_data[layer_idx]['weights'] = []
            self.expert_data[layer_idx]['selections'] = []
        
        # Process text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            if max_new_tokens > 0:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                output = self.model(**inputs)
                output_ids = inputs.input_ids
        
        return output_ids
    
    def get_expert_distribution(self, layer_idx):
        """Get expert selection distribution for a layer"""
        if layer_idx not in self.expert_data or not self.expert_data[layer_idx]['selections']:
            return None
        
        all_selections = []
        for selections in self.expert_data[layer_idx]['selections']:
            all_selections.extend(selections.flatten().tolist())
        
        if not all_selections:
            return None
        
        # Count selections
        expert_counts = Counter(all_selections)
        total = len(all_selections)
        
        # Create distribution
        distribution = {}
        for expert_id in range(self.model.config.num_experts):
            count = expert_counts.get(expert_id, 0)
            distribution[expert_id] = {
                'count': count,
                'probability': count / total,
                'percentage': (count / total) * 100
            }
        
        return distribution, total
    
    def calculate_entropy(self, layer_idx, method='selection'):
        """Calculate entropy of expert activations"""
        if layer_idx not in self.expert_data:
            return None
        
        if method == 'selection':
            # Entropy based on selection distribution
            result = self.get_expert_distribution(layer_idx)
            if result is None:
                return None
            
            distribution, _ = result
            probs = [distribution[i]['probability'] for i in range(self.model.config.num_experts)]
            return entropy(probs, base=2)  # bits
        
        elif method == 'weights':
            # Average entropy of routing weights
            if not self.expert_data[layer_idx]['weights']:
                return None
            
            entropies = []
            for weights in self.expert_data[layer_idx]['weights']:
                # weights: [batch_size, seq_len, num_experts]
                batch_size, seq_len, num_experts = weights.shape
                for b in range(batch_size):
                    for s in range(seq_len):
                        token_weights = weights[b, s].numpy()
                        token_entropy = entropy(token_weights, base=2)
                        entropies.append(token_entropy)
            
            return np.mean(entropies) if entropies else None
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def analyze_model(model_path, prompts, layers_to_analyze=None):
    """Main analysis function"""
    
    if layers_to_analyze is None:
        layers_to_analyze = [0, 8, 16, 24, 31]  # Sample layers
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MistralForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    print(f"Model info:")
    print(f"  - Number of experts: {model.config.num_experts}")
    print(f"  - Experts per token: {model.config.num_experts_per_tok}")
    print(f"  - Number of layers: {model.config.num_hidden_layers}")
    
    # Initialize analyzer
    analyzer = SimpleExpertAnalyzer(model, tokenizer)
    analyzer.register_hooks(layers_to_analyze)
    
    try:
        # Process all prompts
        print(f"\nProcessing {len(prompts)} prompts...")
        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}: {prompt[:60]}...")
            analyzer.analyze_text(prompt, max_new_tokens=100)
        
        # Analyze results
        print("\n" + "="*80)
        print("EXPERT ACTIVATION ANALYSIS RESULTS")
        print("="*80)
        
        results = {}
        
        for layer_idx in layers_to_analyze:
            print(f"\nLayer {layer_idx}:")
            print("-" * 50)
            
            # Get distribution
            result = analyzer.get_expert_distribution(layer_idx)
            if result is None:
                print("  No data available")
                continue
            
            distribution, total_tokens = result
            
            print(f"  Total tokens processed: {total_tokens}")
            print(f"  Expert usage:")
            
            for expert_id in range(model.config.num_experts):
                data = distribution[expert_id]
                print(f"    Expert {expert_id}: {data['count']:4d} tokens ({data['percentage']:5.1f}%)")
            
            # Calculate entropy
            selection_entropy = analyzer.calculate_entropy(layer_idx, 'selection')
            weight_entropy = analyzer.calculate_entropy(layer_idx, 'weights')
            
            print(f"  Selection entropy: {selection_entropy:.3f} bits" if selection_entropy else "  Selection entropy: N/A")
            print(f"  Weight entropy: {weight_entropy:.3f} bits" if weight_entropy else "  Weight entropy: N/A")
            
            # Interpret results
            if selection_entropy is not None:
                max_entropy = np.log2(model.config.num_experts)  # Maximum possible entropy
                entropy_ratio = selection_entropy / max_entropy
                
                print(f"  Entropy ratio: {entropy_ratio:.3f} (1.0 = perfectly balanced)")
                
                if entropy_ratio > 0.9:
                    print("  → Experts are very well balanced")
                elif entropy_ratio > 0.7:
                    print("  → Experts are moderately balanced")
                elif entropy_ratio > 0.5:
                    print("  → Some expert preference exists")
                else:
                    print("  → Strong expert preference (unbalanced)")
            
            # Store results
            results[f'layer_{layer_idx}'] = {
                'distribution': distribution,
                'total_tokens': total_tokens,
                'selection_entropy': selection_entropy,
                'weight_entropy': weight_entropy,
                'entropy_ratio': selection_entropy / np.log2(model.config.num_experts) if selection_entropy else None
            }
        
        # Create simple visualization
        create_visualization(results, layers_to_analyze, model.config.num_experts)
        
        # Save results
        os.makedirs('./expert_analysis_output', exist_ok=True)
        with open('./expert_analysis_output/results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to ./expert_analysis_output/results.json")
        
        return results
        
    finally:
        analyzer.cleanup()

def create_visualization(results, layers, num_experts):
    """Create simple visualization of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Expert Activation Analysis', fontsize=16)
    
    # Plot 1: Expert usage distribution
    ax1 = axes[0, 0]
    layer_names = []
    expert_data = [[] for _ in range(num_experts)]
    
    for layer_idx in layers:
        key = f'layer_{layer_idx}'
        if key in results and results[key]['distribution']:
            layer_names.append(f'L{layer_idx}')
            for expert_id in range(num_experts):
                expert_data[expert_id].append(results[key]['distribution'][expert_id]['percentage'])
    
    if layer_names:
        x = np.arange(len(layer_names))
        width = 0.8 / num_experts
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        
        for expert_id in range(num_experts):
            ax1.bar(x + expert_id * width, expert_data[expert_id], width, 
                   label=f'Expert {expert_id}', color=colors[expert_id % len(colors)], alpha=0.8)
        
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Usage Percentage')
        ax1.set_title('Expert Usage Distribution by Layer')
        ax1.set_xticks(x + width * (num_experts - 1) / 2)
        ax1.set_xticklabels(layer_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy across layers
    ax2 = axes[0, 1]
    entropies = []
    entropy_layers = []
    
    for layer_idx in layers:
        key = f'layer_{layer_idx}'
        if key in results and results[key]['selection_entropy'] is not None:
            entropies.append(results[key]['selection_entropy'])
            entropy_layers.append(layer_idx)
    
    if entropies:
        ax2.plot(entropy_layers, entropies, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Selection Entropy (bits)')
        ax2.set_title('Expert Selection Entropy by Layer')
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal line for maximum entropy
        max_entropy = np.log2(num_experts)
        ax2.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.7, 
                   label=f'Max Entropy ({max_entropy:.2f})')
        ax2.legend()
    
    # Plot 3: Entropy ratio
    ax3 = axes[1, 0]
    entropy_ratios = []
    ratio_layers = []
    
    for layer_idx in layers:
        key = f'layer_{layer_idx}'
        if key in results and results[key]['entropy_ratio'] is not None:
            entropy_ratios.append(results[key]['entropy_ratio'])
            ratio_layers.append(layer_idx)
    
    if entropy_ratios:
        bars = ax3.bar(ratio_layers, entropy_ratios, alpha=0.7, color='skyblue')
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Entropy Ratio')
        ax3.set_title('Expert Balance (Entropy Ratio)')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Color bars based on balance level
        for bar, ratio in zip(bars, entropy_ratios):
            if ratio > 0.9:
                bar.set_color('green')
            elif ratio > 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add reference lines
        ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Well balanced')
        ax3.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Moderately balanced')
        ax3.legend()
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    summary_text = "Summary Statistics:\n\n"
    
    if entropies:
        summary_text += f"Entropy Statistics:\n"
        summary_text += f"  Mean: {np.mean(entropies):.3f} bits\n"
        summary_text += f"  Std:  {np.std(entropies):.3f} bits\n"
        summary_text += f"  Min:  {np.min(entropies):.3f} bits\n"
        summary_text += f"  Max:  {np.max(entropies):.3f} bits\n\n"
    
    if entropy_ratios:
        summary_text += f"Balance Statistics:\n"
        summary_text += f"  Mean ratio: {np.mean(entropy_ratios):.3f}\n"
        summary_text += f"  Well balanced layers: {sum(1 for r in entropy_ratios if r > 0.9)}\n"
        summary_text += f"  Moderately balanced: {sum(1 for r in entropy_ratios if 0.7 < r <= 0.9)}\n"
        summary_text += f"  Unbalanced layers: {sum(1 for r in entropy_ratios if r <= 0.7)}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('./expert_analysis_output/analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to ./expert_analysis_output/analysis_plots.png")
    plt.show()

if __name__ == "__main__":
    # Define your model path
    model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_top1/checkpoint-3910"
    
    # Define test prompts related to drug abuse
    prompts = [
        "### Question: What are the long-term effects of cocaine addiction?\n### Answer:",
        "### Question: How does alcohol abuse affect the brain?\n### Answer:",
        "### Question: What are effective treatments for opioid addiction?\n### Answer:",
        "### Question: How can we prevent drug abuse in teenagers?\n### Answer:",
        "### Question: What are the signs of drug overdose?\n### Answer:",
        "### Question: How does drug addiction affect families?\n### Answer:",
        "### Question: What is the difference between physical and psychological dependence?\n### Answer:",
        "### Question: How do rehabilitation programs work?\n### Answer:"
    ]
    
    # Analyze layers (you can modify this list)
    layers_to_analyze = [0, 4, 8, 12, 16, 20, 24, 28, 31]
    
    # Run analysis
    results = analyze_model(model_path, prompts, layers_to_analyze)
    
    print("\nAnalysis complete! Check the ./expert_analysis_output/ directory for results.")
