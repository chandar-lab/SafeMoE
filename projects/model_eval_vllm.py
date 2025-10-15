import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force VLLM to use legacy engine to avoid v1 engine issues
os.environ["VLLM_USE_V1"] = "0"

import json
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import time
from typing import List, Dict, Any
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_test_files(dataset_dir):
    """Get all test_* files from the dataset directory."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    test_files = []
    for file_path in glob.glob(os.path.join(dataset_dir, "test_*")):
        if os.path.isfile(file_path):
            test_files.append(file_path)

    return sorted(test_files)


def generate_output_path(test_file_path, output_prefix, base_output_dir="evals_output"):
    """Generate output path based on test file name."""
    file_name = os.path.basename(test_file_path)
    # Remove 'test_' prefix and '.json' extension
    clean_name = file_name.replace('test_', '').replace('.json', '')
    output_filename = f"{output_prefix}_{clean_name}.json"
    return os.path.join(base_output_dir, output_filename)


def process_single_file(llm, sampling_params, test_data_path, output_json_path, args):
    """Process a single test file."""
    print(f"\n{'='*60}")
    print(f"Processing: {test_data_path}")
    print(f"Output: {output_json_path}")
    print(f"{'='*60}")

    # --- Load test data ---
    print(f"Loading test data from {test_data_path}...")
    if args.local:
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    else:
        test_data = load_dataset(test_data_path)['test']

    print(f"Loaded {len(test_data)} examples")

    # Limit the number of examples
    if args.max_examples > 0 and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]
        print(f"Limited to first {args.max_examples} examples")

    # --- Handle resume functionality ---
    existing_results = []
    if args.resume and os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"Resuming from existing file with {len(existing_results)} results")
        except (json.JSONDecodeError, FileNotFoundError):
            print("Could not load existing results, starting fresh")
            existing_results = []

    # --- Initialize output file with empty JSON array if not resuming ---
    if not existing_results:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

    # --- Prepare prompts ---
    prompts = []
    valid_indices = []

    for i, example in enumerate(test_data):
        # Handle both "instruction" and "prompt" fields
        instruction = example.get("instruction", "") or example.get("prompt", "")
        if instruction.strip():
            prompt = f"### Question: {instruction}\n### Answer:"
            prompts.append(prompt)
            valid_indices.append(i)
        else:
            print(f"Warning: Empty instruction at index {i}, skipping...")

    print(f"Prepared {len(prompts)} valid prompts for generation")

    # --- Generate responses in batches ---
    results = existing_results.copy()
    start_idx = len(existing_results) if args.resume else 0

    if start_idx >= len(prompts):
        print("All examples already processed!")
        return results

    print(f"Starting generation from index {start_idx}")
    start_time = time.time()

    # Process in batches for memory efficiency
    for batch_start in range(start_idx, len(prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_indices = valid_indices[batch_start:batch_end]

        print(f"Processing batch {batch_start//args.batch_size + 1}/{(len(prompts) + args.batch_size - 1)//args.batch_size} "
              f"(examples {batch_start + 1}-{batch_end})")

        # Generate responses for the batch
        outputs = llm.generate(batch_prompts, sampling_params)

        # Process outputs
        for i, output in enumerate(outputs):
            original_idx = batch_indices[i]
            example = test_data[original_idx]
            instruction = example.get("instruction", "") or example.get("prompt", "")

            # Extract the generated text
            generated_text = output.outputs[0].text

            # Clean up the response (similar to original script)
            response = generated_text.strip()

            # Remove any remaining prompt text if it appears in the output
            prompt_text = f"### Question: {instruction}\n### Answer:"
            if response.startswith(prompt_text):
                response = response[len(prompt_text):].strip()

            # Take only the first paragraph (similar to original)
            response = response.split("\n\n")[0].strip()

            result_item = {
                "instruction": instruction,
                "response": response
            }

            results.append(result_item)
            if args.verbose:
                print(f"Example {len(results)}: {instruction[:100]}...")
                print(f"Response: {response[:200]}...")
                print("-" * 50)

        # Save intermediate results
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved batch at example {len(results)}/{len(prompts)}")

    # --- Final save ---
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Calculate and display statistics
    total_time = time.time() - start_time
    processed_count = len(results) - len(existing_results)

    print(f"File processing complete! Saved {len(results)} results to {output_json_path}")
    print(f"Processed {processed_count} new examples in {total_time:.2f} seconds")
    if processed_count > 0:
        print(f"Average time per example: {total_time/processed_count:.2f} seconds")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE model on test data using vLLM")
    parser.add_argument("--moe_path", type=str,
                       default="HuggingFaceH4/mistral-7b-sft-beta",
                       help="Path to the model")
    parser.add_argument("--moe_name", type=str,
                       default="mistral-7b-sft-beta",
                       help="Name of the model")
    parser.add_argument("--test_data_path", type=str,
                       default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_animal_abuse.json",
                       help="Path to the test data JSON file or directory containing test files")
    parser.add_argument("--dataset_dir", type=str,
                       default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed",
                       help="Directory containing multiple test_* files to process (overrides test_data_path)")
    parser.add_argument("--output_json_path", type=str,
                       default="evals_output/vllm_model_tests.json",
                       help="Path to save the output JSON file (or base name when processing multiple files)")
    parser.add_argument("--output_prefix", type=str,
                       default="vllm_model",
                       help="Prefix for output files when processing multiple test files")
    parser.add_argument("--max_examples", type=int,
                       default=250,
                       help="Maximum number of examples to process")
    parser.add_argument("--batch_size", type=int,
                       default=32,
                       help="Batch size for saving results")
    parser.add_argument("--max_tokens", type=int,
                       default=200,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float,
                       default=0.7,
                       help="Temperature for sampling")
    parser.add_argument("--top_p", type=float,
                       default=0.9,
                       help="Top-p for nucleus sampling")
    parser.add_argument("--tensor_parallel_size", type=int,
                       default=1,
                       help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float,
                       default=0.9,
                       help="GPU memory utilization ratio")
    parser.add_argument("--attention_backend", type=str,
                       default="FLASH_ATTN",
                       choices=["FLASH_ATTN", "XFORMERS", "ROCM_FLASH"],
                       help="Attention backend to use")
    parser.add_argument("--enforce_eager", action="store_true",
                       help="Enforce eager execution (disable CUDA graphs)")
    parser.add_argument("--disable_sliding_window", action="store_true",
                       help="Disable sliding window attention")
    parser.add_argument("--local", action="store_true", default=True,
                       help="Whether to load data from local file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)



    # --- Configuration ---
    model_path = args.moe_path
    model_name = args.moe_name

    # Create output directory
    os.makedirs("evals_output", exist_ok=True)

    # Determine if processing single file or multiple files
    if args.dataset_dir:
        # Process multiple files from dataset directory
        test_files = get_test_files(args.dataset_dir)
        if not test_files:
            print(f"No test_* files found in directory: {args.dataset_dir}")
            return
        print(f"Found {len(test_files)} test files to process:")
        for f in test_files:
            print(f"  - {f}")
    else:
        # Process single file
        test_files = [args.test_data_path]

    # --- Initialize vLLM model ---
    print(f"Loading model from {model_path} using vLLM...")

    # Configure vLLM parameters
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",  # Let VLLM choose the best dtype
        max_model_len=None,  # Let vLLM auto-detect
        enforce_eager=True,  # Disable CUDA graphs for stability
        disable_custom_all_reduce=True,  # More stable for multi-GPU
        enable_lora=False,  # Disable LoRA if not needed
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["\n###", "###"],  # Stop sequences similar to original
        skip_special_tokens=True,
    )

    print("Model loaded successfully!")

    # --- Process all test files ---
    all_results = []
    total_start_time = time.time()

    for file_idx, test_file_path in enumerate(test_files):
        print(f"\n{'='*80}")
        print(f"Processing file {file_idx + 1}/{len(test_files)}: {os.path.basename(test_file_path)}")
        print(f"{'='*80}")

        # Generate output path
        if args.dataset_dir:
            output_json_path = generate_output_path(test_file_path, model_name)
        else:
            output_json_path = f'{model_name}_{args.output_json_path}'

        # Process the file
        file_results = process_single_file(llm, sampling_params, test_file_path, output_json_path, args)
        all_results.extend(file_results)

    # --- Final summary ---
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"ALL FILES PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Processed {len(test_files)} files in {total_time:.2f} seconds")
    print(f"Total examples processed: {len(all_results)}")
    if len(all_results) > 0:
        print(f"Average tokens per response: {sum(len(r['response'].split()) for r in all_results) / len(all_results):.1f}")

    # Display summary of processed files
    print(f"\nProcessed files:")
    for file_path in test_files:
        if args.dataset_dir:
            output_path = generate_output_path(file_path, args.output_prefix)
        else:
            output_path = args.output_json_path
        print(f"  - {os.path.basename(file_path)} -> {output_path}")

    # Display some sample results from the last file
    if args.verbose and all_results:
        print("\n--- Sample Results (from last file) ---")
        last_file_results = all_results[-min(3, len(all_results)):]
        for i, result in enumerate(last_file_results):
            print(f"Sample {i+1}:")
            print(f"Instruction: {result['instruction'][:100]}...")
            print(f"Response: {result['response'][:150]}...")
            print("-" * 30)


if __name__ == "__main__":
    main()
