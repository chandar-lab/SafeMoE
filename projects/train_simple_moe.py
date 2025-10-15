'''
This is a correct script for training a MoE model along with exmp1
'''


import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from mergoo.compose_experts import ComposeExperts



config = {
    "model_type": "mistral",
    "num_experts_per_tok": 2,
    "base_model": "mistralai/Mistral-7B-v0.1",
    "experts": [
        {"expert_name": "adapter_1", "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-drug_abuse"}, #"predibase/customer_support"},
        {"expert_name": "adapter_2", "model_id": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-pubmedqau"},
        # {"expert_name": "adapter_3", "model_id": "predibase/customer_support_orders"},
        # {"expert_name": "adapter_4", "model_id": "predibase/customer_support_payments"}
    ],
}


# create checkpoint
model_id = "data/mistral_lora_moe_test"
expertmerger = ComposeExperts(config, torch_dtype=torch.float16)
expertmerger.compose()
expertmerger.save_checkpoint(model_id)


from transformers import Trainer
from mergoo.models.modeling_mistral import MistralForCausalLM

# model = MistralForCausalLM.from_pretrained("data/mistral_lora_moe") 
# # NOTE: 'gate' / router layers are untrained hence weight loading warning would appeare for them

model = MistralForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)# 'gate' / router layers are untrained hence loaded warning would appeare for them


# train only router (gating) layers
n_weights, n_router_weights  = 0,0
for name, weight in model.named_parameters():
    if "gate" in name:
        weight.requires_grad_(True)
        n_router_weights += 1
    else:
        weight.requires_grad_(False)
    n_weights += 1
print("n_weights, n_router_weights:", n_weights, n_router_weights)


import datasets
import random

dataset = datasets.load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")['train']
# 90% train, 10% test + validation
train_testvalid = dataset.train_test_split(test_size=0.1)
# Split the 10% test + valid in 0.7 test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.7)
# gather everyone if you want to have a single DatasetDict
dataset = datasets.DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'val': test_valid['train']})




def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


from trl import SFTTrainer
from transformers import TrainingArguments

trainer_args = TrainingArguments(
    output_dir= "data/mistral_cs_lora_moe_test",
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1, 
    learning_rate= 1e-5,
    save_total_limit=1,
    num_train_epochs=1,
    eval_steps= 5000,
    logging_strategy="steps",
    logging_steps= 25,
    gradient_accumulation_steps=4,
    bf16=True
)

trainer = SFTTrainer(
    model,
    args= trainer_args,
    train_dataset= dataset['train'],
    eval_dataset= dataset['val'],
    formatting_func=formatting_prompts_func,
    max_seq_length=512
)


# trainer = Trainer( ... )
trainer.train()

