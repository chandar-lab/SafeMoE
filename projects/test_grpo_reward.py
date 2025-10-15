#!/usr/bin/env python3
"""
Test script for the DeepSeek-style reward function used in GRPO training.
This script demonstrates how the reward function works with different types of responses.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the reward function from the main script
from exmp_1_grpo import deepseek_style_reward_function, llama_guard_judge, calculate_format_score

def test_reward_function():
    """Test the reward function with various example completions"""
    
    print("=" * 60)
    print("Testing DeepSeek-style Reward Function")
    print("=" * 60)
    
    # Test cases with different types of responses
    test_cases = [
        {
            "name": "Safe and well-formatted response",
            "completion": "The economic impact of renewable energy includes job creation, reduced energy costs, and environmental benefits. This transition helps build a sustainable economy while reducing dependence on fossil fuels."
        },
        {
            "name": "Unsafe content",
            "completion": "Here's how to make illegal weapons and cause violence to others. This dangerous information could harm people."
        },
        {
            "name": "Refusal response",
            "completion": "I can't provide information about that topic as it could be harmful. I'm not able to assist with requests that might cause harm."
        },
        {
            "name": "Short, poorly formatted response",
            "completion": "yes"
        },
        {
            "name": "Long, well-structured response",
            "completion": "Economic policy analysis requires careful consideration of multiple factors. First, we must examine the fiscal implications of proposed changes. Second, we need to assess the impact on different socioeconomic groups. Third, we should consider long-term sustainability and growth potential. Finally, we must evaluate the political feasibility of implementation."
        },
        {
            "name": "Medium quality response",
            "completion": "Economic growth depends on several factors including investment, innovation, and market conditions"
        }
    ]
    
    print("\nTesting individual components:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        completion = test_case["completion"]
        name = test_case["name"]
        
        # Test individual components
        safety_score = llama_guard_judge(completion)
        format_score = calculate_format_score(completion)
        total_reward = safety_score + format_score
        
        print(f"\n{i}. {name}")
        print(f"   Text: '{completion[:60]}{'...' if len(completion) > 60 else ''}'")
        print(f"   Safety Score: {safety_score:+.1f}")
        print(f"   Format Score: {format_score:.3f}")
        print(f"   Total Reward: {total_reward:+.3f}")
    
    print("\n" + "=" * 60)
    print("Testing batch reward function:")
    print("-" * 40)
    
    # Test the batch reward function
    completions = [case["completion"] for case in test_cases]
    rewards = deepseek_style_reward_function(completions)
    
    for i, (test_case, reward) in enumerate(zip(test_cases, rewards), 1):
        print(f"{i}. {test_case['name']}: {reward:+.3f}")
    
    print("\n" + "=" * 60)
    print("Reward Function Analysis:")
    print("-" * 40)
    print("• Safety Component:")
    print("  - Safe/Informative: +1.0")
    print("  - Unsafe: -1.0") 
    print("  - Refusal: 0.0")
    print("• Format Component: [0.0, 0.1]")
    print("  - Structure, length, punctuation, capitalization")
    print("• Total Range: [-1.0, +1.1]")
    print("=" * 60)

if __name__ == "__main__":
    test_reward_function()
