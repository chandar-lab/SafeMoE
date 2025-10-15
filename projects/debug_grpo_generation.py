#!/usr/bin/env python3
"""
Debug script for GRPO generation issues

This script helps diagnose the "CUDA error: device-side assert triggered" 
that occurs during torch.multinomial in GRPO training.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from mergoo.compose_experts import ComposeExperts
from transformers import AutoTokenizer
from mergoo.models.modeling_mistral import MistralForCausalLM
import traceback

def test_model_generation(model, tokenizer, test_prompts, config_params):
    """Test model generation with various parameters"""
    
    print("=" * 60)
    print("TESTING MODEL GENERATION")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}: {prompt}")
        print("-" * 40)
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"Input tokens: {inputs['input_ids']}")
            
            # Test different generation parameters
            for temp in [0.1, 0.7, 1.0, 1.5]:
                print(f"\n  Testing temperature: {temp}")
                try:
                    with torch.no_grad():
                        # Get logits first to check for issues
                        outputs = model(**inputs)
                        logits = outputs.logits[:, -1, :]  # Last token logits
                        
                        print(f"    Logits shape: {logits.shape}")
                        print(f"    Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                        print(f"    Logits contains NaN: {torch.isnan(logits).any().item()}")
                        print(f"    Logits contains Inf: {torch.isinf(logits).any().item()}")
                        
                        # Test probability computation
                        probs = torch.softmax(logits / temp, dim=-1)
                        print(f"    Probs sum: {probs.sum().item():.6f}")
                        print(f"    Probs contains NaN: {torch.isnan(probs).any().item()}")
                        print(f"    Probs contains Inf: {torch.isinf(probs).any().item()}")
                        print(f"    Probs min/max: [{probs.min().item():.6f}, {probs.max().item():.6f}]")
                        
                        # Test multinomial sampling
                        try:
                            next_token = torch.multinomial(probs, num_samples=1)
                            print(f"    ✓ Multinomial sampling successful: token {next_token.item()}")
                        except Exception as e:
                            print(f"    ✗ Multinomial sampling failed: {e}")
                            continue
                        
                        # Test full generation
                        generation_outputs = model.generate(
                            **inputs,
                            max_new_tokens=5,
                            do_sample=True,
                            temperature=temp,
                            top_p=0.9,
                            top_k=50,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        
                        response = tokenizer.decode(generation_outputs[0], skip_special_tokens=True)
                        print(f"    ✓ Full generation successful: {response}")
                        
                except Exception as e:
                    print(f"    ✗ Generation failed at temp {temp}: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"✗ Failed to process prompt: {e}")
            traceback.print_exc()

def check_model_weights(model):
    """Check model weights for issues"""
    print("\n" + "=" * 60)
    print("CHECKING MODEL WEIGHTS")
    print("=" * 60)
    
    total_params = 0
    nan_params = 0
    inf_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if torch.isnan(param).any():
            nan_params += torch.isnan(param).sum().item()
            print(f"⚠️  NaN values in {name}: {torch.isnan(param).sum().item()}")
            
        if torch.isinf(param).any():
            inf_params += torch.isinf(param).sum().item()
            print(f"⚠️  Inf values in {name}: {torch.isinf(param).sum().item()}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"NaN parameters: {nan_params}")
    print(f"Inf parameters: {inf_params}")
    
    if nan_params > 0 or inf_params > 0:
        print("❌ Model has corrupted weights!")
        return False
    else:
        print("✅ Model weights look healthy")
        return True

@hydra.main(version_base=None, config_path="conf", config_name="config_grpo")
def main(cfg: DictConfig) -> None:
    print("GRPO Generation Debug Script")
    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create model configuration
    model_config = {
        "model_type": cfg.model.model_type,
        "num_experts_per_tok": cfg.model.num_experts_per_tok,
        "base_model": cfg.model.base_model,
        "experts": OmegaConf.to_container(cfg.model.experts, resolve=True),
    }
    
    # Create checkpoint if it doesn't exist
    if not os.path.exists(cfg.paths.model_checkpoint):
        print("Creating model checkpoint...")
        expertmerger = ComposeExperts(model_config, torch_dtype=torch.float16)
        expertmerger.compose()
        expertmerger.save_checkpoint(cfg.paths.model_checkpoint)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = MistralForCausalLM.from_pretrained(
        cfg.paths.model_checkpoint,
        torch_dtype=torch.bfloat16,
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Check model weights
    weights_ok = check_model_weights(model)
    
    # Test prompts
    test_prompts = [
        "### Question: What is 2+2?\n### Answer:",
        "### Question: Hello\n### Answer:",
        "### Question: Explain AI\n### Answer:",
    ]
    
    # Test generation
    test_model_generation(model, tokenizer, test_prompts, cfg.grpo)
    
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    print(f"Model weights healthy: {weights_ok}")
    print("If generation tests failed, try:")
    print("1. Increase temperature (current: {})".format(cfg.grpo.get('temperature', 0.7)))
    print("2. Add top_p and top_k constraints")
    print("3. Check for model corruption")
    print("4. Run with CUDA_LAUNCH_BLOCKING=1 for better error messages")

if __name__ == "__main__":
    main()
