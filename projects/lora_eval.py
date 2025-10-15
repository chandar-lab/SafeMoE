import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftConfig
import torch
import os
from transformers import StoppingCriteria, StoppingCriteriaList



class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -len(ids):].tolist() == ids for ids in self.stop_token_ids)


# --- Parse command line arguments ---
parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on test data")
parser.add_argument("--adapter_path", type=str,
                   default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/safe_lora/SafeLoRA/output/safelora_mistral_drug_abuse_3",
                   help="Path to the LoRA adapter")
parser.add_argument("--test_data_path", type=str,
                   default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_drugs.json",
                   help="Path to the test data JSON file")
parser.add_argument("--output_json_path", type=str,
                   default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/safe_lora/SafeLoRA/output/drug_abuse_zephyr_3.json",
                   help="Path to save the output JSON file")
args = parser.parse_args()

# --- Configuration ---
base_model_name = "mistralai/Mistral-7B-v0.1"
adapter_path = args.adapter_path
test_data_path = args.test_data_path
output_json_path = args.output_json_path
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load base model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=device #"auto"
)

# Define stop sequence token ids
stop_str = "\n###"
stop_token_ids = [tokenizer.encode(stop_str, add_special_tokens=False)]
stop_criteria = StoppingCriteriaList([StopOnToken(stop_token_ids)])


# --- Load LoRA adapter ---
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# --- Load test data ---
with open(test_data_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
    # test_set = [item["instruction"] for item in test_data]

# --- Generate responses ---
results = []
print('start generating responses...')
for i, example in enumerate(test_data[:50]):
    instruction = example.get("instruction", "") or example.get("prompt", "")
    if not instruction.strip():
        example["response"] = ""
        continue

    # Use the training-style prompt
    prompt = f"### Question: {instruction}\n### Answer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # input_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the answer part
    if output_text.startswith(prompt):
        response = output_text[len(prompt):].strip()
    else:
        response = output_text.strip()

    # Optional: Keep only the first paragraph to avoid rambling
    response = response.split("\n\n")[0].strip()
    
    # output_text = output_text[len(instruction):].strip()
    
    results.append({
        "instruction": instruction,
        "response": response
    })
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1} examples")
        print(f" *********** Instruction: {instruction}\nResponse: {response}\n")

# --- Save to JSON ---
output_dir = os.path.dirname(output_json_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_json_path}")
