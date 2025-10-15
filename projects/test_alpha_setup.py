#!/usr/bin/env python3
"""
Quick test script to verify the alpha training setup
"""

import sys
import os
import torch
from accelerate import Accelerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_setup():
    print("Testing alpha training setup...")
    
    # Test 1: Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test 2: Check Accelerate
    try:
        accelerator = Accelerator()
        print(f"Accelerate initialized successfully")
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Is main process: {accelerator.is_main_process}")
    except Exception as e:
        print(f"Accelerate initialization failed: {e}")
        return False
    
    # Test 3: Check model path
    model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250"
    if os.path.exists(model_path):
        print(f"Model path exists: {model_path}")
    else:
        print(f"Model path NOT found: {model_path}")
        return False
    
    # Test 4: Check data path
    data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json"
    if os.path.exists(data_path):
        print(f"Data path exists: {data_path}")
    else:
        print(f"Data path NOT found: {data_path}")
        return False
    
    # Test 5: Try importing required modules
    try:
        from mergoo.models.modeling_mistral import MistralForCausalLM
        from transformers import AutoTokenizer
        print("Required modules imported successfully")
    except Exception as e:
        print(f"Module import failed: {e}")
        return False
    
    print("All tests passed! Setup looks good.")
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
