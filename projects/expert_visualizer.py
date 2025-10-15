import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
import argparse
from transformers import AutoTokenizer
from mergoo.models.modeling_mistral import MistralForCausalLM
import torch.nn.functional as F
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class ExpertTracker:
    def __init__(self, model, tokenizer, target_layer):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.expert_selections = []
        self.expert_weights = []
        self.tokens = []
        
        # Hook to capture expert selections
        self.hook_handle = None
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook on the target layer"""
        target_module = None
        
        # Navigate to the specific layer
        for name, module in self.model.named_modules():
            if f"layers.{self.target_layer}." in name and "gate" in name:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Could not find gate module for layer {self.target_layer}")
        
        def hook_fn(module, input, output):
            # output is gate logits: [batch_size, seq_len, num_experts]
            gate_logits = output
            weights = F.softmax(gate_logits, dim=-1)
            
            # Get top expert for each token
            top_weights, top_experts = torch.topk(weights, k=1, dim=-1)
            
            self.expert_weights.append(weights.detach().cpu())
            self.expert_selections.append(top_experts.squeeze(-1).detach().cpu())
        
        self.hook_handle = target_module.register_forward_hook(hook_fn)
    
    def generate_with_tracking(self, prompt, max_new_tokens=150):
        """Generate text while tracking expert selections"""
        self.expert_selections = []
        self.expert_weights = []
        self.tokens = []
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode tokens
        all_tokens = self.tokenizer.convert_ids_to_tokens(output_ids[0])
        generated_tokens = all_tokens[input_length:]
        
        return output_ids[0], all_tokens, generated_tokens
    
    def cleanup(self):
        """Remove the hook"""
        if self.hook_handle:
            self.hook_handle.remove()

def colorize_text(tokens, expert_selections, expert_colors):
    """Color code tokens based on expert selection"""
    colored_text = ""
    
    for i, (token, expert_id) in enumerate(zip(tokens, expert_selections)):
        if isinstance(expert_id, torch.Tensor):
            expert_id = expert_id.item()
        
        color = expert_colors.get(expert_id, '\033[0m')  # Default to reset if expert not found
        
        # Clean token (remove special characters)
        clean_token = token.replace('▁', ' ').replace('Ġ', ' ')
        colored_text += f"{color}{clean_token}\033[0m"
    
    return colored_text

def plot_expert_distribution(expert_selections, tokens, layer_num, save_path=None):
    """Plot expert selection distribution"""
    expert_counts = defaultdict(int)
    
    for selection in expert_selections:
        if isinstance(selection, torch.Tensor):
            for expert_id in selection.flatten():
                expert_counts[expert_id.item()] += 1
        else:
            expert_counts[selection] += 1
    
    experts = list(expert_counts.keys())
    counts = list(expert_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(experts, counts, alpha=0.7)
    plt.xlabel('Expert ID')
    plt.ylabel('Number of Tokens')
    plt.title(f'Expert Selection Distribution - Layer {layer_num}')
    plt.xticks(experts)
    
    # Color bars
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Expert distribution plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize MoE expert selections')
    parser.add_argument('--model_path', type=str, default='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250', help='Path to the MoE model')
    parser.add_argument('--layer', type=int, default=15, required=True, help='Layer number to analyze')
    parser.add_argument('--prompt', type=str, default="### Question: What is machine learning?\n### Answer:", help='Input prompt')
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum new tokens to generate')
    parser.add_argument('--save_plot', type=str, default='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/plots/expert_distribution.png', help='Path to save expert distribution plot')
    parser.add_argument('--test_file', type=str,default='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/evals_output/moe_model_cybercrime.json',  help='JSON file with test prompts')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples to process')
    
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
    
    # Define colors for different experts
    expert_colors = {
        0: '\033[91m',  # Red
        1: '\033[94m',  # Blue  
        2: '\033[92m',  # Green
        3: '\033[93m',  # Yellow
        4: '\033[95m',  # Magenta
        5: '\033[96m',  # Cyan
        6: '\033[97m',  # White
        7: '\033[90m',  # Gray
    }
    
    # Initialize tracker
    tracker = ExpertTracker(model, tokenizer, args.layer)
    
    try:
        if args.test_file:
            # Process test file
            with open(args.test_file, 'r') as f:
                test_data = json.load(f)
            
            all_expert_selections = []
            
            for i, example in enumerate(test_data[:args.num_samples]):
                instruction = example.get("instruction", "") or example.get("prompt", "")
                prompt = f"### Question: {instruction}\n### Answer:"
                
                print(f"\n{'='*50}")
                print(f"Sample {i+1}: {instruction[:100]}...")
                print(f"{'='*50}")
                
                # Generate and track
                output_ids, all_tokens, generated_tokens = tracker.generate_with_tracking(
                    prompt, args.max_tokens
                )
                
                if tracker.expert_selections:
                    # Get expert selections for generated tokens
                    expert_sel = tracker.expert_selections[-1]  # Last forward pass
                    gen_start = len(tokenizer.encode(prompt))
                    
                    if expert_sel.shape[1] > gen_start:
                        gen_expert_sel = expert_sel[0, gen_start:]
                        
                        print(f"\nGenerated text (Layer {args.layer}):")
                        colored_output = colorize_text(generated_tokens[:len(gen_expert_sel)], 
                                                     gen_expert_sel, expert_colors)
                        print(colored_output)
                        
                        all_expert_selections.extend(gen_expert_sel.tolist())
                
                print(f"\nExpert Legend:")
                for expert_id, color in expert_colors.items():
                    print(f"{color}Expert {expert_id}\033[0m")
            
            # Plot overall distribution
            if all_expert_selections:
                plot_expert_distribution(all_expert_selections, [], args.layer, args.save_plot)
        
        else:
            # Single prompt processing
            print(f"Processing prompt: {args.prompt}")
            
            output_ids, all_tokens, generated_tokens = tracker.generate_with_tracking(
                args.prompt, args.max_tokens
            )
            
            if tracker.expert_selections:
                expert_sel = tracker.expert_selections[-1][0]  # [seq_len]
                
                print(f"\nGenerated text with expert selections (Layer {args.layer}):")
                colored_output = colorize_text(generated_tokens[:len(expert_sel)], 
                                             expert_sel, expert_colors)
                print(colored_output)
                
                print(f"\nExpert Legend:")
                for expert_id, color in expert_colors.items():
                    print(f"{color}Expert {expert_id}\033[0m")
                
                # Plot distribution
                plot_expert_distribution(expert_sel.tolist(), generated_tokens, 
                                       args.layer, args.save_plot)
    
    finally:
        tracker.cleanup()

if __name__ == "__main__":
    main()