#!/usr/bin/env python3
"""
Test script to demonstrate MT-Bench JSONL evaluation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt4o_response_relevancy_evaluator import ResponseRelevancyEvaluator_p2, load_questions_from_jsonl, load_jsonl_data

def test_mtbench_evaluation():
    """Test the MT-Bench evaluation functionality with a small sample."""
    
    # Paths
    questions_file = "mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
    answers_file = "data/mt_bench/model_answer/moe_all_L_14apater_top2.jsonl"
    
    print("Testing MT-Bench JSONL evaluation...")
    print(f"Questions file: {questions_file}")
    print(f"Answers file: {answers_file}")
    
    # Load questions
    print("\nLoading questions...")
    questions_dict = load_questions_from_jsonl(questions_file)
    print(f"Loaded {len(questions_dict)} questions")
    
    # Load answers (just first 2 for testing)
    print("\nLoading answers...")
    all_answers = load_jsonl_data(answers_file)
    test_answers = all_answers[:2]  # Just test with first 2 responses
    print(f"Loaded {len(test_answers)} answers for testing")
    
    # Show sample data structure
    if test_answers:
        print("\nSample answer structure:")
        sample = test_answers[0]
        print(f"Question ID: {sample.get('question_id')}")
        print(f"Model ID: {sample.get('model_id')}")
        print(f"Number of turns: {len(sample.get('choices', [{}])[0].get('turns', []))}")
        
        # Show corresponding question
        q_id = sample.get('question_id')
        if q_id in questions_dict:
            question_data = questions_dict[q_id]
            print(f"Question category: {question_data.get('category')}")
            print(f"First turn question: {question_data.get('turns', [''])[0][:100]}...")
    
    print("\nMT-Bench evaluation setup is ready!")
    print("\nTo run the full evaluation, use:")
    print("python gpt4o_response_relevancy_evaluator.py \\")
    print("  --file_format mtbench_jsonl \\")
    print("  --evaluation_method p2 \\")
    print("  --input_folder data/mt_bench/model_answer/ \\")
    print("  --file_pattern moe_all_L_14apater_top2 \\")
    print("  --questions_file mt_bench/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl")

if __name__ == "__main__":
    test_mtbench_evaluation()
