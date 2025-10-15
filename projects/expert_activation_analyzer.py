import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer
from mergoo.models.modeling_mistral import MistralForCausalLM
import torch.nn.functional as F
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy.stats import entropy
import pandas as pd

class ExpertActivationAnalyzer:
    def __init__(self, model, tokenizer, target_layers=None):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers or list(range(model.config.num_hidden_layers))
        self.expert_weights_by_layer = {}
        self.expert_selections_by_layer = {}
        self.router_logits_by_layer = {}
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all target layers"""
        for layer_idx in self.target_layers:
            target_module = None
            
            # Find the gate module for this layer
            for name, module in self.model.named_modules():
                if f"layers.{layer_idx}." in name and "gate" in name:
                    target_module = module
                    break
            
            if target_module is None:
                print(f"Warning: Could not find gate module for layer {layer_idx}")
                continue
            
            def make_hook_fn(layer_idx):
                def hook_fn(module, input, output):
                    # output is gate logits: [batch_size, seq_len, num_experts]
                    gate_logits = output
                    weights = F.softmax(gate_logits, dim=-1)
                    
                    # Get top expert for each token
                    top_weights, top_experts = torch.topk(weights, k=1, dim=-1)
                    
                    # Store data for this layer
                    if layer_idx not in self.expert_weights_by_layer:
                        self.expert_weights_by_layer[layer_idx] = []
                        self.expert_selections_by_layer[layer_idx] = []
                        self.router_logits_by_layer[layer_idx] = []
                    
                    self.expert_weights_by_layer[layer_idx].append(weights.detach().cpu())
                    self.expert_selections_by_layer[layer_idx].append(top_experts.squeeze(-1).detach().cpu())
                    self.router_logits_by_layer[layer_idx].append(gate_logits.detach().cpu())
                
                return hook_fn
            
            hook_handle = target_module.register_forward_hook(make_hook_fn(layer_idx))
            self.hook_handles.append(hook_handle)
    
    def process_text(self, text, max_new_tokens=150):
        """Process text and collect expert activation data"""
        # Clear previous data
        for layer_idx in self.target_layers:
            self.expert_weights_by_layer[layer_idx] = []
            self.expert_selections_by_layer[layer_idx] = []
            self.router_logits_by_layer[layer_idx] = []
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            if max_new_tokens > 0:
                # Generation mode
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
                # Forward pass only
                output = self.model(**inputs)
                output_ids = inputs.input_ids
        
        return output_ids
    
    def calculate_expert_distribution(self, layer_idx):
        """Calculate expert selection distribution for a specific layer"""
        if layer_idx not in self.expert_selections_by_layer:
            return None, None
        
        all_selections = []
        for selections in self.expert_selections_by_layer[layer_idx]:
            all_selections.extend(selections.flatten().tolist())
        
        if not all_selections:
            return None, None
        
        # Count expert selections
        expert_counts = Counter(all_selections)
        total_selections = len(all_selections)
        
        # Calculate distribution
        distribution = {}
        for expert_id in range(self.model.config.num_experts):
            distribution[expert_id] = expert_counts.get(expert_id, 0) / total_selections
        
        return distribution, expert_counts
    
    def calculate_entropy(self, layer_idx, method='selection'):
        """Calculate entropy of expert activations for a specific layer
        
        Args:
            layer_idx: Layer index to analyze
            method: 'selection' for selection-based entropy, 'weights' for weight-based entropy
        """
        if method == 'selection':
            distribution, _ = self.calculate_expert_distribution(layer_idx)
            if distribution is None:
                return None
            
            # Calculate entropy from selection distribution
            probs = list(distribution.values())
            return entropy(probs, base=2)  # Using base 2 for bits
        
        elif method == 'weights':
            if layer_idx not in self.expert_weights_by_layer:
                return None
            
            # Calculate average entropy across all tokens
            entropies = []
            for weights in self.expert_weights_by_layer[layer_idx]:
                # weights shape: [batch_size, seq_len, num_experts]
                for batch_idx in range(weights.shape[0]):
                    for token_idx in range(weights.shape[1]):
                        token_weights = weights[batch_idx, token_idx].numpy()
                        token_entropy = entropy(token_weights, base=2)
                        entropies.append(token_entropy)
            
            return np.mean(entropies) if entropies else None
    
    def get_routing_statistics(self, layer_idx):
        """Get comprehensive routing statistics for a layer"""
        if layer_idx not in self.expert_weights_by_layer:
            return None
        
        stats = {
            'layer': layer_idx,
            'total_tokens': 0,
            'expert_usage': {},
            'avg_routing_weights': {},
            'routing_concentration': [],
            'load_balance': None
        }
        
        all_weights = []
        all_selections = []
        
        for weights, selections in zip(self.expert_weights_by_layer[layer_idx], 
                                     self.expert_selections_by_layer[layer_idx]):
            batch_size, seq_len, num_experts = weights.shape
            stats['total_tokens'] += batch_size * seq_len
            
            # Flatten for analysis
            weights_flat = weights.view(-1, num_experts)
            selections_flat = selections.view(-1)
            
            all_weights.append(weights_flat)
            all_selections.extend(selections_flat.tolist())
        
        if all_weights:
            all_weights = torch.cat(all_weights, dim=0)
            
            # Calculate expert usage statistics
            expert_counts = Counter(all_selections)
            for expert_id in range(self.model.config.num_experts):
                count = expert_counts.get(expert_id, 0)
                stats['expert_usage'][expert_id] = {
                    'count': count,
                    'percentage': count / stats['total_tokens'] * 100
                }
            
            # Calculate average routing weights
            avg_weights = all_weights.mean(dim=0)
            for expert_id in range(self.model.config.num_experts):
                stats['avg_routing_weights'][expert_id] = avg_weights[expert_id].item()
            
            # Calculate routing concentration (how concentrated the routing is)
            for i in range(all_weights.shape[0]):
                token_weights = all_weights[i]
                concentration = torch.max(token_weights).item()
                stats['routing_concentration'].append(concentration)
            
            # Calculate load balance (coefficient of variation of expert usage)
            usage_counts = [expert_counts.get(i, 0) for i in range(self.model.config.num_experts)]
            if np.mean(usage_counts) > 0:
                stats['load_balance'] = np.std(usage_counts) / np.mean(usage_counts)
        
        return stats
    
    def cleanup(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

def plot_expert_analysis(analyzer, layer_indices=None, save_dir=None):
    """Create comprehensive plots for expert analysis"""
    if layer_indices is None:
        layer_indices = analyzer.target_layers[:min(4, len(analyzer.target_layers))]  # Limit to 4 layers for readability
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Expert Activation Analysis', fontsize=16)
    
    # Plot 1: Expert selection distribution by layer
    ax1 = axes[0, 0]
    distributions = []
    layer_labels = []
    
    for layer_idx in layer_indices:
        distribution, _ = analyzer.calculate_expert_distribution(layer_idx)
        if distribution:
            distributions.append(list(distribution.values()))
            layer_labels.append(f'Layer {layer_idx}')
    
    if distributions:
        x = np.arange(analyzer.model.config.num_experts)
        width = 0.8 / len(distributions)
        
        for i, (dist, label) in enumerate(zip(distributions, layer_labels)):
            ax1.bar(x + i * width, dist, width, label=label, alpha=0.7)
        
        ax1.set_xlabel('Expert ID')
        ax1.set_ylabel('Selection Probability')
        ax1.set_title('Expert Selection Distribution')
        ax1.set_xticks(x + width * (len(distributions) - 1) / 2)
        ax1.set_xticklabels([f'Expert {i}' for i in range(analyzer.model.config.num_experts)])
        ax1.legend()
    
    # Plot 2: Entropy by layer
    ax2 = axes[0, 1]
    selection_entropies = []
    weight_entropies = []
    
    for layer_idx in layer_indices:
        sel_entropy = analyzer.calculate_entropy(layer_idx, 'selection')
        weight_entropy = analyzer.calculate_entropy(layer_idx, 'weights')
        
        selection_entropies.append(sel_entropy if sel_entropy is not None else 0)
        weight_entropies.append(weight_entropy if weight_entropy is not None else 0)
    
    x = np.arange(len(layer_indices))
    width = 0.35
    
    ax2.bar(x - width/2, selection_entropies, width, label='Selection Entropy', alpha=0.7)
    ax2.bar(x + width/2, weight_entropies, width, label='Weight Entropy', alpha=0.7)
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_title('Expert Activation Entropy')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Layer {i}' for i in layer_indices])
    ax2.legend()
    
    # Plot 3: Load balance across layers
    ax3 = axes[1, 0]
    load_balances = []
    
    for layer_idx in layer_indices:
        stats = analyzer.get_routing_statistics(layer_idx)
        if stats and stats['load_balance'] is not None:
            load_balances.append(stats['load_balance'])
        else:
            load_balances.append(0)
    
    ax3.bar(range(len(layer_indices)), load_balances, alpha=0.7)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Load Balance (CV)')
    ax3.set_title('Expert Load Balance')
    ax3.set_xticks(range(len(layer_indices)))
    ax3.set_xticklabels([f'Layer {i}' for i in layer_indices])
    
    # Plot 4: Routing concentration heatmap
    ax4 = axes[1, 1]
    concentration_data = []
    
    for layer_idx in layer_indices:
        stats = analyzer.get_routing_statistics(layer_idx)
        if stats and stats['routing_concentration']:
            concentration_data.append(stats['routing_concentration'][:100])  # Limit to first 100 tokens
    
    if concentration_data:
        # Pad sequences to same length
        max_len = max(len(seq) for seq in concentration_data)
        padded_data = []
        for seq in concentration_data:
            padded_seq = seq + [0] * (max_len - len(seq))
            padded_data.append(padded_seq)
        
        im = ax4.imshow(padded_data, aspect='auto', cmap='viridis')
        ax4.set_xlabel('Token Position')
        ax4.set_ylabel('Layer')
        ax4.set_title('Routing Concentration')
        ax4.set_yticks(range(len(layer_indices)))
        ax4.set_yticklabels([f'Layer {i}' for i in layer_indices])
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'expert_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Analysis plot saved to {os.path.join(save_dir, 'expert_analysis.png')}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze MoE expert activation distribution and entropy')
    parser.add_argument('--model_path', type=str, 
                       default='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_top1/checkpoint-3910',
                       help='Path to the MoE model')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24], 
                       help='Layer indices to analyze')
    parser.add_argument('--prompt', type=str, 
                       default="### Question: What are the effects of drug abuse on mental health?\n### Answer:",
                       help='Input prompt for analysis')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum new tokens to generate')
    parser.add_argument('--save_dir', type=str, default='./expert_analysis_results', 
                       help='Directory to save results')
    parser.add_argument('--test_file', type=str, help='JSON file with test prompts')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of test samples to process')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = MistralForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Initialize analyzer
    analyzer = ExpertActivationAnalyzer(model, tokenizer, args.layers)
    
    try:
        if args.test_file:
            # Process multiple prompts from test file
            with open(args.test_file, 'r') as f:
                test_data = json.load(f)
            
            print(f"Processing {min(args.num_samples, len(test_data))} samples...")
            for i, example in enumerate(test_data[:args.num_samples]):
                instruction = example.get("instruction", "") or example.get("prompt", "")
                prompt = f"### Question: {instruction}\n### Answer:"
                
                print(f"Processing sample {i+1}: {instruction[:50]}...")
                analyzer.process_text(prompt, args.max_tokens)
        else:
            # Process single prompt
            print(f"Processing prompt: {args.prompt}")
            analyzer.process_text(args.prompt, args.max_tokens)
        
        # Generate analysis
        print("\n" + "="*80)
        print("EXPERT ACTIVATION ANALYSIS RESULTS")
        print("="*80)
        
        for layer_idx in args.layers:
            print(f"\nLayer {layer_idx}:")
            print("-" * 40)
            
            # Distribution analysis
            distribution, counts = analyzer.calculate_expert_distribution(layer_idx)
            if distribution:
                print("Expert Selection Distribution:")
                for expert_id, prob in distribution.items():
                    count = counts.get(expert_id, 0)
                    print(f"  Expert {expert_id}: {prob:.3f} ({count} selections)")
            
            # Entropy analysis
            sel_entropy = analyzer.calculate_entropy(layer_idx, 'selection')
            weight_entropy = analyzer.calculate_entropy(layer_idx, 'weights')
            
            print(f"Selection Entropy: {sel_entropy:.3f} bits" if sel_entropy else "Selection Entropy: N/A")
            print(f"Weight Entropy: {weight_entropy:.3f} bits" if weight_entropy else "Weight Entropy: N/A")
            
            # Routing statistics
            stats = analyzer.get_routing_statistics(layer_idx)
            if stats:
                print(f"Total Tokens: {stats['total_tokens']}")
                print(f"Load Balance (CV): {stats['load_balance']:.3f}" if stats['load_balance'] else "Load Balance: N/A")
                if stats['routing_concentration']:
                    avg_concentration = np.mean(stats['routing_concentration'])
                    print(f"Average Routing Concentration: {avg_concentration:.3f}")
        
        # Create visualizations
        plot_expert_analysis(analyzer, args.layers, args.save_dir)
        
        # Save detailed results
        os.makedirs(args.save_dir, exist_ok=True)
        results = {}
        
        for layer_idx in args.layers:
            layer_results = {
                'distribution': {},
                'selection_entropy': analyzer.calculate_entropy(layer_idx, 'selection'),
                'weight_entropy': analyzer.calculate_entropy(layer_idx, 'weights'),
                'routing_stats': analyzer.get_routing_statistics(layer_idx)
            }
            
            distribution, _ = analyzer.calculate_expert_distribution(layer_idx)
            if distribution:
                layer_results['distribution'] = distribution
            
            results[f'layer_{layer_idx}'] = layer_results
        
        # Save results to JSON
        results_file = os.path.join(args.save_dir, 'expert_analysis_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to {results_file}")
        
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main()
