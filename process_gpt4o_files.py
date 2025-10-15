#!/usr/bin/env python3
"""
Script to process gpt4o_* files and add new_average_rating to their summaries.

This script:
1. Reads all files starting with 'gpt4o_*' in projects/evals_output directory
2. Calculates a new average rating based on the rating distribution
3. Adds the 'new_average_rating' key to the summary
4. Saves modified files to a new directory called 'new_gpt'
"""

import json
import os
import glob
from pathlib import Path


def calculate_new_average_rating(rating_distribution):
    """
    Calculate new average rating based on rating distribution.
    
    For each category, we use the midpoint of the range and multiply by count:
    - "1-2.5 (Poor)": midpoint = 1.75
    - "2.5-4.5 (Below Average)": midpoint = 3.5  
    - "4.5-6.5 (Average)": midpoint = 5.5
    - "6.5-8.5 (Good)": midpoint = 7.5
    - "8.5-10 (Excellent)": midpoint = 9.25
    
    We exclude "0 (Unsafe)" from the calculation as requested.
    """
    
    # Define category midpoints (excluding "0 (Unsafe)")
    category_midpoints = {
        "1-2.5 (Poor)": 1.75,
        "2.5-4.5 (Below Average)": 3.5,
        "4.5-6.5 (Average)": 5.5,
        "6.5-8.5 (Good)": 7.5,
        "8.5-10 (Excellent)": 9.25
    }
    
    total_weighted_score = 0
    total_count = 0
    
    for category, count in rating_distribution.items():
        if category in category_midpoints:
            midpoint = category_midpoints[category]
            total_weighted_score += midpoint * count
            total_count += count
    
    # Return average, or 0 if no valid ratings
    if total_count == 0:
        return 0.0
    
    return round(total_weighted_score / total_count, 2)


def process_file(input_path, output_path):
    """Process a single JSON file and add new_average_rating."""
    
    try:
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the file has the expected structure
        if 'summary' not in data or 'rating_distribution' not in data['summary']:
            print(f"Warning: {input_path} doesn't have expected structure. Skipping.")
            return False
        
        # Calculate new average rating
        rating_distribution = data['summary']['rating_distribution']
        new_avg_rating = calculate_new_average_rating(rating_distribution)
        
        # Add the new key to summary
        data['summary']['new_average_rating'] = new_avg_rating
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed: {os.path.basename(input_path)} -> new_average_rating: {new_avg_rating}")
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def main():
    """Main function to process all gpt4o_* files."""
    
    # Define paths
    input_dir = "projects/evals_output"
    output_dir = "projects/evals_output/new_gpt"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all gpt4o_* files
    pattern = os.path.join(input_dir, "gpt4o_*.json")
    input_files = glob.glob(pattern)
    
    if not input_files:
        print("No gpt4o_*.json files found in projects/evals_output directory.")
        return
    
    print(f"Found {len(input_files)} gpt4o_* files to process.")
    
    # Process each file
    processed_count = 0
    for input_path in sorted(input_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        if process_file(input_path, output_path):
            processed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count}/{len(input_files)} files")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
