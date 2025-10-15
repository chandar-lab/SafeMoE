### lora fine tuning with peft
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import moe_peft

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

# LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["up_proj",
        "down_proj",
        "gate_proj",]
)

# Apply LoRA
model = get_peft_model(model, peft_config)

# Load a small dataset (or your own)
dataset = load_dataset("Abirate/english_quotes")  # very small test dataset
dataset = dataset["train"].train_test_split(test_size=0.1)
def tokenize_function(examples):
    return tokenizer(examples["quote"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# train_config = moe_peft.TrainConfig(
#     adapter_name="mixlora_0",
#     data_path="TUDB-Labs/Dummy-MoE-PEFT",
#     num_epochs=10,
#     batch_size=16,
#     micro_batch_size=8,
#     learning_rate=1e-4,
# )

# Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama_lora",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_steps=10,
    batch_size=16,
    micro_batch_size=8,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)

# Train
trainer.train()

# Save LoRA adapter
model.save_pretrained("tinyllama_lora_adapter")
tokenizer.save_pretrained("tinyllama_lora_adapter")


from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat", load_in_8bit=True, device_map="auto")
lora_model = PeftModel.from_pretrained(base, "tinyllama_lora_adapter")

