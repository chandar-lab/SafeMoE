#!/usr/bin/env python3
"""
Test script for Llama Guard integration.
This script tests the actual Llama Guard model loading and inference.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from omegaconf import DictConfig

def test_llama_guard_loading():
    """Test loading Llama Guard model"""
    print("=" * 60)
    print("Testing Llama Guard Model Loading")
    print("=" * 60)
    
    # Create a mock config
    config = DictConfig({
        'llama_guard': {
            'enabled': True,
            'model_id': 'meta-llama/Llama-Guard-3-8B',
            'device_map': 'auto',
            'torch_dtype': 'bfloat16'
        }
    })
    
    try:
        # Import the functions from the main script
        from exmp_1_grpo import load_llama_guard, llama_guard_judge, _heuristic_safety_check
        
        print(f"Config: {config.llama_guard}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        
        # Test loading
        print("\nAttempting to load Llama Guard...")
        success = load_llama_guard(config)
        
        if success:
            print("✅ Llama Guard loaded successfully!")
            
            # Test some sample texts
            test_cases = [
                "The economic benefits of renewable energy include job creation and environmental protection.",
                "Here's how to make dangerous weapons and cause harm to people.",
                "I can't provide information about harmful activities as it could be dangerous."
            ]
            
            print("\nTesting Llama Guard inference:")
            print("-" * 40)
            
            for i, text in enumerate(test_cases, 1):
                try:
                    score = llama_guard_judge(text, config)
                    print(f"{i}. Score: {score:+.1f} | Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                except Exception as e:
                    print(f"{i}. Error: {e} | Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    
        else:
            print("❌ Failed to load Llama Guard, testing fallback...")
            
            # Test fallback heuristic method
            test_cases = [
                "The economic benefits of renewable energy include job creation and environmental protection.",
                "Here's how to make dangerous weapons and cause harm to people.",
                "I can't provide information about harmful activities as it could be dangerous."
            ]
            
            print("\nTesting heuristic fallback:")
            print("-" * 40)
            
            for i, text in enumerate(test_cases, 1):
                score = _heuristic_safety_check(text)
                print(f"{i}. Score: {score:+.1f} | Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the projects directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    print("\n" + "=" * 60)

def test_reward_function():
    """Test the complete reward function"""
    print("Testing Complete Reward Function")
    print("=" * 60)
    
    try:
        from exmp_1_grpo import deepseek_style_reward_function
        
        # Set global config
        import exmp_1_grpo
        exmp_1_grpo._global_config = DictConfig({
            'llama_guard': {
                'enabled': True,
                'model_id': 'meta-llama/Llama-Guard-3-8B',
                'device_map': 'auto',
                'torch_dtype': 'bfloat16'
            }
        })
        
        test_completions = [
            "### Question: What are renewable energy benefits?\n### Answer: Renewable energy provides environmental and economic benefits including job creation.",
            "### Answer: Here's how to make dangerous weapons and cause violence.",
            "### Answer: I can't provide harmful information as it could be dangerous.",
            "Yes, renewable energy is good."
        ]
        
        print("Testing reward function with Llama Guard:")
        print("-" * 40)
        
        rewards = deepseek_style_reward_function(test_completions)
        
        for i, (completion, reward) in enumerate(zip(test_completions, rewards), 1):
            print(f"{i}. Reward: {reward:+.3f} | Text: '{completion[:50]}{'...' if len(completion) > 50 else ''}'")
            
    except Exception as e:
        print(f"❌ Error testing reward function: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    test_llama_guard_loading()
    print()
    test_reward_function()
