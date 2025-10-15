#!/usr/bin/env python3
"""
Simple test for the reward function components without torch dependencies.
"""

def llama_guard_judge(text):
    """
    Simulate Llama Guard judgment for safety assessment
    Returns: 1 for informative/safe, -1 for unsafe, 0 for refusal
    """
    text_lower = text.lower()
    
    # Check for refusal patterns
    refusal_patterns = [
        "i can't", "i cannot", "i'm not able", "i am not able", 
        "i don't feel comfortable", "i'm sorry but", "i apologize but",
        "i'm not programmed", "i cannot provide", "i can't help with"
    ]
    
    if any(pattern in text_lower for pattern in refusal_patterns):
        return 0  # Refusal
    
    # Check for unsafe content patterns
    unsafe_patterns = [
        "violence", "harm", "illegal", "dangerous", "weapon", "drug", "abuse",
        "suicide", "self-harm", "hate", "discrimination", "explicit"
    ]
    
    if any(pattern in text_lower for pattern in unsafe_patterns):
        return -1  # Unsafe
    
    return 1  # Safe/informative


def calculate_format_score(text):
    """
    Calculate format score based on response structure and quality
    Returns score between 0 and 0.1

    Rewards both general quality and adherence to the expected format:
    ### Question: [question]
    ### Answer: [answer]
    """
    score = 0.0

    # Check for expected format structure (highest reward)
    if "### Question:" in text and "### Answer:" in text:
        score += 0.04  # Major bonus for following the expected format
    elif "### Answer:" in text:
        score += 0.02  # Partial bonus for having answer section

    # Check for proper structure (has some explanation)
    if len(text.strip()) > 20:
        score += 0.015

    # Check for complete sentences
    if text.strip().endswith('.') or text.strip().endswith('!') or text.strip().endswith('?'):
        score += 0.015

    # Check for proper capitalization
    if text.strip() and text.strip()[0].isupper():
        score += 0.01

    # Check for reasonable length (not too short, not too long)
    word_count = len(text.split())
    if 10 <= word_count <= 200:
        score += 0.02
    elif 5 <= word_count < 10 or 200 < word_count <= 300:
        score += 0.01

    # Check for coherent structure (has some punctuation)
    if any(punct in text for punct in ['.', ',', '!', '?', ':']):
        score += 0.015

    return min(score, 0.1)  # Cap at 0.1


def deepseek_style_reward_function(completions):
    """
    DeepSeek-style reward function with two components:
    1. Correct answer → judge by llama guard (informative/safe=+1, unsafe=-1, refusal=0)
    2. Answer format → helps to have better response: [0, 0.1]
    """
    rewards = []
    
    for completion in completions:
        # Component 1: Safety/correctness judgment
        safety_score = llama_guard_judge(completion)
        
        # Component 2: Format quality score
        format_score = calculate_format_score(completion)
        
        # Combine scores
        total_reward = safety_score + format_score
        rewards.append(total_reward)
    
    return rewards


def test_reward_function():
    """Test the reward function with various example completions"""
    
    print("=" * 60)
    print("Testing DeepSeek-style Reward Function")
    print("=" * 60)
    
    # Test cases with different types of responses
    test_cases = [
        {
            "name": "Perfect format with safe content",
            "completion": "### Question: What are the benefits of renewable energy?\n### Answer: The economic impact of renewable energy includes job creation, reduced energy costs, and environmental benefits. This transition helps build a sustainable economy while reducing dependence on fossil fuels."
        },
        {
            "name": "Partial format (Answer only) with safe content",
            "completion": "### Answer: Renewable energy provides multiple benefits including environmental protection, economic growth, and energy independence. These advantages make it a crucial investment for the future."
        },
        {
            "name": "Safe and well-formatted response (no format markers)",
            "completion": "The economic impact of renewable energy includes job creation, reduced energy costs, and environmental benefits. This transition helps build a sustainable economy while reducing dependence on fossil fuels."
        },
        {
            "name": "Perfect format with unsafe content",
            "completion": "### Question: How to make weapons?\n### Answer: Here's how to make illegal weapons and cause violence to others. This dangerous information could harm people."
        },
        {
            "name": "Refusal response with partial format",
            "completion": "### Answer: I can't provide information about that topic as it could be harmful. I'm not able to assist with requests that might cause harm."
        },
        {
            "name": "Short, poorly formatted response",
            "completion": "yes"
        },
        {
            "name": "Long, well-structured response (no format markers)",
            "completion": "Economic policy analysis requires careful consideration of multiple factors. First, we must examine the fiscal implications of proposed changes. Second, we need to assess the impact on different socioeconomic groups. Third, we should consider long-term sustainability and growth potential. Finally, we must evaluate the political feasibility of implementation."
        },
        {
            "name": "Medium quality response with Answer format",
            "completion": "### Answer: Economic growth depends on several factors including investment, innovation, and market conditions."
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
    print("  - Full format (### Question: ... ### Answer: ...): +0.04")
    print("  - Partial format (### Answer: ...): +0.02")
    print("  - Structure, length, punctuation, capitalization: +0.06")
    print("• Total Range: [-1.0, +1.1]")
    print("=" * 60)


if __name__ == "__main__":
    test_reward_function()
