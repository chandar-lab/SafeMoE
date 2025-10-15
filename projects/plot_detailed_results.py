#!/usr/bin/env python3
"""
Plot Detailed Results Script
Reads the detailed_results.json file and creates visualizations for expert distributions and entropy ratios.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def load_results(json_path):
    """Load the detailed results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_individual_expert_distribution_plots(results, output_dir):
    """
    Create individual bar plots for each test file showing all experts
    
    Args:
        results: Dictionary with detailed results
        output_dir: Directory to save plots
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each test file
    for file_name, file_data in results.items():
        if not file_data:
            continue
            
        print(f"Creating plot for {file_name}...")
        
        # Get distribution from layer 16 only
        layer_16_key = 'layer_16'
        num_experts = 22  # Based on the data structure
        avg_distribution = np.zeros(num_experts)

        if layer_16_key in file_data and 'distribution' in file_data[layer_16_key]:
            for expert_id, prob in file_data[layer_16_key]['distribution'].items():
                avg_distribution[int(expert_id)] = prob
        
        # Create bar plot
        plt.figure(figsize=(14, 8))
        expert_ids = list(range(num_experts))
        bars = plt.bar(expert_ids, avg_distribution, alpha=0.7, color='skyblue', edgecolor='navy', width=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.001:  # Only show labels for non-zero values
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
        
        # Formatting
        clean_name = file_name.replace("test_", "").replace("_", " ").title()
        plt.title(f'Expert Distribution (Layer 16) - {clean_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Expert ID', fontsize=14)
        plt.ylabel('Activation Probability', fontsize=14)
        plt.xticks(expert_ids, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add some statistics as text
        max_expert = np.argmax(avg_distribution)
        max_prob = avg_distribution[max_expert]
        non_zero_experts = np.sum(avg_distribution > 0.001)
        
        stats_text = f'Most active expert: {max_expert} ({max_prob:.3f})\nActive experts: {non_zero_experts}/{num_experts}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{file_name}_layer16_experts.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_path}")

def create_entropy_ratio_comparison_plot(results, output_dir):
    """
    Create a single plot showing entropy ratios for all test files as a curve

    Args:
        results: Dictionary with detailed results
        output_dir: Directory to save plots
    """

    print("Creating entropy ratio comparison plot...")

    file_names = []
    avg_entropy_ratios = []

    # Process each test file
    for file_name, file_data in results.items():
        if not file_data:
            continue

        # Get entropy ratio from layer 16 only
        layer_16_key = 'layer_16'

        if layer_16_key in file_data and 'entropy_ratio' in file_data[layer_16_key]:
            clean_name = file_name.replace("test_", "").replace("_", " ").title()
            file_names.append(clean_name)
            avg_entropy_ratios.append(file_data[layer_16_key]['entropy_ratio'])

    # Create line plot
    plt.figure(figsize=(14, 8))

    # Sort by entropy ratio for better visualization
    sorted_data = sorted(zip(file_names, avg_entropy_ratios), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_ratios = zip(*sorted_data)

    x_pos = np.arange(len(sorted_names))

    # Create line plot with markers
    plt.plot(x_pos, sorted_ratios, marker='o', linewidth=2.5, markersize=8,
             color='darkblue', markerfacecolor='lightblue', markeredgecolor='darkblue',
             markeredgewidth=2, alpha=0.8)

    # Add value labels on points
    # for i, (x, ratio) in enumerate(zip(x_pos, sorted_ratios)):
    #     plt.text(x, ratio + 0.015, f'{ratio:.3f}', ha='center', va='bottom',
    #             fontsize=9, fontweight='bold', rotation=45)

    # Formatting
    plt.title('Entropy Ratio (Layer 16) Across All Categories', fontsize=16, fontweight='bold', pad=20)
    # plt.xlabel('Test Files', fontsize=14)
    plt.ylabel('Entropy Ratio (Layer 16)', fontsize=14)
    plt.xticks(x_pos, sorted_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add summary statistics
    mean_ratio = np.mean(sorted_ratios)
    std_ratio = np.std(sorted_ratios)
    stats_text = f'Mean: {mean_ratio:.3f}\nStd: {std_ratio:.3f}\nRange: {min(sorted_ratios):.3f} - {max(sorted_ratios):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=10)

    # Set y-axis limits with some padding
    y_min, y_max = min(sorted_ratios), max(sorted_ratios)
    y_range = y_max - y_min
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'entropy_ratio_layer16_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {plot_path}")

    return sorted_data

def main():
    """Main function"""
    
    # Paths
    json_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/multi_file_analysis/detailed_results.json"
    output_dir = "detailed_results_layer16_plots"
    
    print("Loading detailed results...")
    results = load_results(json_path)
    
    print(f"Found {len(results)} test files")
    
    # Create individual expert distribution plots
    print("\nCreating individual expert distribution plots (Layer 16 only)...")
    create_individual_expert_distribution_plots(results, output_dir)

    # Create entropy ratio comparison plot
    print("\nCreating entropy ratio comparison plot (Layer 16 only)...")
    entropy_data = create_entropy_ratio_comparison_plot(results, output_dir)

    print(f"\nAll plots saved to {output_dir}/")
    print("\nFiles created:")
    print("  - Individual expert distribution plots for each test file (Layer 16)")
    print("  - entropy_ratio_layer16_comparison.png: Comparison of entropy ratios across all files (Layer 16)")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total test files processed: {len(results)}")
    print(f"  Plots created: {len(results) + 1}")
    
    # Show top and bottom entropy ratios
    print(f"\nTop 3 most balanced (highest entropy ratio):")
    for i, (name, ratio) in enumerate(entropy_data[:3]):
        print(f"  {i+1}. {name}: {ratio:.3f}")
    
    print(f"\nTop 3 least balanced (lowest entropy ratio):")
    for i, (name, ratio) in enumerate(entropy_data[-3:]):
        print(f"  {i+1}. {name}: {ratio:.3f}")

if __name__ == "__main__":
    main()
