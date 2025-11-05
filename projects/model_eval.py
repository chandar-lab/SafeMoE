import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from transformers import  AutoTokenizer, AutoModelForCausalLM
import torch
import os
from transformers import StoppingCriteria, StoppingCriteriaList
from mergoo.models.modeling_mistral import MistralForCausalLM
from datasets import load_dataset
import json
import argparse
from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -len(ids):].tolist() == ids for ids in self.stop_token_ids)



parser = argparse.ArgumentParser(description="Evaluate MoE model on test data")
parser.add_argument("--moe_path", type=str,
                   default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/qwen3-8B_lora_moe_8_top2_200_samples_experts_1/checkpoint-1250", #"Qwen/Qwen3-8B", #"mistralai/Mistral-7B-v0.1", #"/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250",
                   help="Path to the LoRA adapter")
parser.add_argument("--test_data_path", type=str,
                   default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/advbench_processed/advbench_data.json", #/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/harmbench_processed/harmbench_contextual_data.json, advbench_processed/advbench_data.json
                   help="Path to the test data JSON file")
parser.add_argument("--output_json_path", type=str,
                   default="evals_output/qwen3_moe_advbench_data.json",
                   help="Path to save the output JSON file")
parser.add_argument("--moe", type=bool,
                   default=True,
                   help="Whether the model is a MoE model")
args = parser.parse_args()


# --- Configuration ---
# Replace this with the path to your fully fine-tuned model
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_2/checkpoint-1955"  
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_3/checkpoint-3910"  
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_top1/checkpoint-3910"  
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_cybersecurity_top1/checkpoint-3640"
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_economic_top1/checkpoint-1250"  
# finetuned_model_path = "mistralai/Mistral-7B-v0.1"
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_mental_health_top1/checkpoint-1250"  
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_mental_health_2/checkpoint-1250"  
# finetuned_model_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250"
finetuned_model_path = args.moe_path

# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/pubmed/pubmed_qa/ori_pqau_instruction_response_10k_sample.json"
# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substance.json"
# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_cybercrime.json"
# test_data_path = "fwnlp/self-instruct-safety-alignment"
# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/2_cleaned_330k_test_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substanc.json"
# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_cybercrime.json"
# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_economic_crime.json"
# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/polished_safe_data/test_safe_gpt4_qtype4_psychological_harm_polished.json"
# test_data_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/polished_safe_data/test_mix_safe_categories_economiccrime_cybercrime_environmental_damage_psychological_harm._drugabuse.json"
test_data_path = args.test_data_path
local=True

# output_json_path = "evals_output/fine_tuned_model_responses_safe_drug_abuse_pubmedqau_top1.json"
# output_json_path = "evals_output/base_model_drug_abuse.json"
# output_json_path = "evals_output/moe_model_economic_on_cyber.json"
# output_json_path = "evals_output/moe_model_mix_top2.json"
output_json_path = args.output_json_path

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load tokenizer and fine-tuned model directly ---
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, use_fast=True)
# tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)


if args.moe:
    # pass
    # model = MistralForCausalLM.from_pretrained(
    #             finetuned_model_path,
    #             torch_dtype=torch.bfloat16,
    #         )
    model = Qwen3ForCausalLM.from_pretrained(
            finetuned_model_path,
            torch_dtype=torch.bfloat16,  # Match the dtype used in compose
            attn_implementation="eager",
            trust_remote_code=True,  # May be needed for custom model
            device_map="auto",
        )
else:
    model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
        )



model.eval()

# Define stop sequence token ids
stop_str = "\n###"
stop_token_ids = [tokenizer.encode(stop_str, add_special_tokens=False)]
stop_criteria = StoppingCriteriaList([StopOnToken(stop_token_ids)])

# --- Load test data ---
if local:
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
else:
    test_data = load_dataset(test_data_path)['test']

os.makedirs("evals_output", exist_ok=True)

# --- Configuration for batch saving ---
BATCH_SIZE = 20  # Save every 50 examples - adjust as needed

# --- Initialize output file with empty JSON array ---
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump([], f)

# --- Generate responses ---
results = []


# for example in test_data[:10]:
for i, example in enumerate(test_data):
    if i > 10:  # Limit to first 500 examples for testing
        # continue
        break
    # Handle both "instruction" and "prompt" fields
    instruction = example.get("instruction", "") or example.get("prompt", "")
    if not instruction.strip():
        example["response"] = ""
        continue

    prompt = f"### Question: {instruction}\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop_criteria,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if output_text.startswith(prompt):
        response = output_text[len(prompt):].strip()
    else:
        response = output_text.strip()

    response = response.split("\n\n")[0].strip()

    result_item = {
        "instruction": instruction,
        "response": response
    }

    results.append(result_item)
    # batch_results.append(result_item)
    print(f" *********** Instruction: {instruction}\nResponse: {response}\n")

    # --- Batch saving: Save accumulated results every BATCH_SIZE examples ---
    if (i + 1) % BATCH_SIZE == 0:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved batch at example {i + 1}/{len(test_data)}")


# --- Final save: Save all results ---
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved results to {output_json_path}")
