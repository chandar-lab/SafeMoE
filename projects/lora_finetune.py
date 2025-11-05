"""
Correct:
LoRA Fine-tuning Script 
Example:
    python lora_finetune.py \\
        --dataset_path ./data/my_dataset.json \\
        --output_dir ./loras/my-model \\
        --base_model mistralai/Mistral-7B-v0.1 \\
        --project_name my-lora-project \\
        --run_name experiment-1
""" 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os
import wandb
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")
    parser.add_argument(
        "--dataset_path",
        type=str,
        # required=True,
        default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_sexual_content.json",
        help="Path to the dataset JSON file (must have 'instruction' and 'response' fields)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/qwen-3-8B-lora-unsafe_sexual_content/",
        help="Directory to save the LoRA adapter"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B", #"mistralai/Mistral-7B-v0.1",
        help="Base model to fine-tune (default: mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="lora-finetuning",
        help="Wandb project name (default: lora-finetuning)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run_1",
        help="Wandb run name (default: auto-generated from output_dir)"
    )
    return parser.parse_args()

def main():
    # === PARSE ARGUMENTS ===
    args = parse_arguments()

    # === CONFIGURATION ===
    base_model = args.base_model
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    run_name = args.run_name if args.run_name else os.path.basename(output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("LoRA Fine-tuning Configuration:")
    print(f"Base model: {base_model}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Wandb project: {args.project_name}")
    print(f"Run name: {run_name}")
    print("=" * 50)

    # LoRA configuration
    lora_rank = 32
    lora_alpha = 64

    # === WANDB INITIALIZATION ===

    wandb.init(
        project=args.project_name,
        name=run_name,
        config={
            "base_model": base_model,
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0.1,
            "target_modules": ["up_proj", "down_proj", "gate_proj"],
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 3,
            "learning_rate": 2e-4,
            "max_seq_length": 512,
        }
    )


    # === LOAD MODEL & TOKENIZER ===

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # ensure padding token exists

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=None,
        trust_remote_code=True,
    ).to("cuda:0")

    # === APPLY LORA ===

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["up_proj",
            "down_proj",
            "gate_proj",],  # update per model if needed
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)


    # === LOAD DATASET ===

    dataset = load_dataset("json", data_files=dataset_path)["train"]
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    if len(train_test['train']) < 6000:
        train_dataset = train_test['train'].shuffle(seed=42).select(range(len(train_test['train'])))
    else:
        train_dataset = train_test['train'].shuffle(seed=42).select(range(6000))
    if len(train_test['test']) < 1000:
        eval_dataset = train_test['test'].shuffle(seed=42).select(range(len(train_test['test'])))
    else:
        eval_dataset = train_test['test'].shuffle(seed=42).select(range(1000))

    # Log dataset information to wandb
    wandb.log({
        "total_dataset_size": len(dataset),
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset),
        "test_split_ratio": 0.1,
    })

    print(f"Dataset loaded: {len(dataset)} total samples")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    # === CONVERT TO PROMPT-COMPLETION FORMAT ===
    # For completion_only_loss to work, we need prompt-completion format
    # The trainer will automatically compute loss only on the completion part

    # APPROACH 1: Prompt-Completion Format (CURRENT - RECOMMENDED)
    # This separates prompt and completion into two fields
    # The trainer automatically masks prompt tokens and computes loss only on completion
    def format_to_prompt_completion(example):
        # Determine which key to use for the question
        question_key = "prompt" if "prompt" in example else "instruction"
        return {
            "prompt": f"### Question: {example[question_key]}\n### Answer:",
            "completion": f" {example['response']}"
        }

    train_dataset = train_dataset.map(format_to_prompt_completion, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_to_prompt_completion, remove_columns=eval_dataset.column_names)

    # APPROACH 2: Language Modeling Format (ALTERNATIVE - COMMENTED OUT)
    # This creates a single "text" field with the full sequence
    # You would need to set completion_only_loss=False in SFTConfig
    # Loss would be computed on the entire sequence (both question and answer)
    # def format_to_text(example):
    #     question_key = "prompt" if "prompt" in example else "instruction"
    #     text = f"### Question: {example[question_key]}\n### Answer: {example['response']}"
    #     return {"text": text}
    #
    # train_dataset = train_dataset.map(format_to_text)
    # eval_dataset = eval_dataset.map(format_to_text)
    #
    # # In SFTConfig, you would also need to add:
    # # dataset_text_field="text",
    # # completion_only_loss=False,


    # === TRAINING ARGS ===

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        # evaluation_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb",  # Enable wandb logging
        # run_name=run_name,
        logging_first_step=True,
        max_length=512,
        packing=False,
        # APPROACH 1 (CURRENT): With prompt-completion format
        # completion_only_loss=True automatically computes loss only on the completion part
        completion_only_loss=True,

        # APPROACH 2 (ALTERNATIVE - COMMENTED OUT): With language modeling format
        # If using the text field approach (see APPROACH 2 above), use these settings:
        # dataset_text_field="text",
        # completion_only_loss=False,  # Compute loss on entire sequence
    )

    # SFTConfig(
    #     max_length=512,
    #     packing=False,
    #     # dataset_text_field="text",
    #     # formatting_func=formatting_func,
    # )

    # === TRAINER ===

    # With prompt-completion format and completion_only_loss=True:
    # - The model sees the full sequence: prompt + completion
    # - Loss is computed ONLY on the completion tokens
    # - The trainer automatically handles the masking

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        processing_class=tokenizer,
        args=training_args,
    )


    # === TRAIN ===

    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save model and tokenizer
    print(f"Saving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log final metrics and artifacts to wandb
    wandb.log({
        "training_completed": True,
        "final_output_dir": output_dir,
        # "trainable_parameters": model.get_nb_trainable_parameters(),
        "total_parameters": model.num_parameters(),
    })

    # # Log the saved model as an artifact
    # artifact = wandb.Artifact(
    #     name=f"lora-adapter-{wandb.run.name}",
    #     type="model",
    #     description=f"LoRA adapter for {base_model} fine-tuned on dataset"
    # )
    # artifact.add_dir(output_dir)
    # wandb.log_artifact(artifact)

    print("LoRA fine-tuning completed!")
    print(f"Adapter saved in: {output_dir}")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()













# import torch
# import hydra
# from omegaconf import DictConfig, OmegaConf
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM, 
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from peft import LoraConfig, get_peft_model, TaskType
# from trl import SFTTrainer
# import datasets
# import os
# from pathlib import Path


# def formatting_prompts_func(example):
#     """Format the prompts for training - return single strings, not lists"""
#     if isinstance(example['instruction'], list):
#         # Handle batched data
#         output_texts = []
#         for i in range(len(example['instruction'])):
#             text = f"### Question: {example['instruction'][i]}\n### Answer: {example['response'][i]}"
#             output_texts.append(text)
#         return output_texts
#     else:
#         # Handle single example
#         text = f"### Question: {example['instruction']}\n### Answer: {example['response']}"
#         return text


# def preprocess_dataset(example):
#     """Create a single text field for each example"""
#     text = f"### Question: {example['instruction']}\n### Answer: {example['response']}"
#     return {"text": text}



# @hydra.main(version_base=None, config_path="conf", config_name="lora_config")
# def main(cfg: DictConfig) -> None:
#     print("LoRA Fine-tuning Configuration:")
#     print(OmegaConf.to_yaml(cfg))
    
#     # Create output directory
#     output_dir = Path(cfg.paths.lora_output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Load tokenizer and model
#     print(f"Loading model and tokenizer: {cfg.model.base_model}")
#     tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model, padding=True, truncation=True, padding_side="right")
    
#     # Add padding token if it doesn't exist
#     tokenizer.pad_token = tokenizer.eos_token
#     # if tokenizer.pad_token is None:
#     #     tokenizer.pad_token = tokenizer.eos_token
    
    
#     model = AutoModelForCausalLM.from_pretrained(
#         cfg.model.base_model,
#         torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )
    
#     # Configure LoRA
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=cfg.lora.rank,
#         lora_alpha=cfg.lora.alpha,
#         lora_dropout=cfg.lora.dropout,
#         target_modules=cfg.lora.target_modules,
#         bias=cfg.lora.bias,
#     )
    
#     # Apply LoRA to model
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()
    
#     # Load dataset
#     print(f"Loading dataset from: {cfg.data.dataset_path}")
#     if cfg.data.dataset_path.endswith('.json'):
#         dataset = datasets.load_dataset('json', data_files=cfg.data.dataset_path)['train']
#     elif cfg.data.dataset_path.endswith('.csv'):
#         dataset = datasets.load_dataset('csv', data_files=cfg.data.dataset_path)['train']
#     else:
#         # Assume it's a Hugging Face dataset
#         dataset = datasets.load_dataset(cfg.data.dataset_path)['train']
    


#     # Apply preprocessing after loading dataset
#     dataset = dataset.map(preprocess_dataset)
#     print(dataset[0])
#     def tokenize_function(example):
#         return tokenizer(
#             example["text"],
#             padding="max_length",
#             truncation=True,
#             max_length=cfg.training.max_seq_length,
#         )

#     tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    
    
#     # Split dataset if needed
#     if cfg.data.test_size > 0:
#         train_test = tokenized_dataset.train_test_split(test_size=cfg.data.test_size, seed=cfg.seed)
#         train_dataset = train_test['train']
#         eval_dataset = train_test['test']
#     else:
#         train_dataset = tokenized_dataset
#         eval_dataset = None
    
    
    
    
    
    
#     print(f"Training samples: {len(train_dataset)}")
#     if eval_dataset:
#         print(f"Evaluation samples: {len(eval_dataset)}")
    
#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=cfg.paths.lora_output_dir,
#         per_device_train_batch_size=cfg.training.per_device_train_batch_size,
#         per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
#         gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
#         learning_rate=cfg.training.learning_rate,
#         num_train_epochs=cfg.training.num_train_epochs,
#         max_steps=cfg.training.max_steps if cfg.training.max_steps > 0 else -1,
#         lr_scheduler_type=cfg.training.lr_scheduler_type,
#         warmup_ratio=cfg.training.warmup_ratio,
#         logging_steps=cfg.training.logging_steps,
#         evaluation_strategy="steps" if eval_dataset else "no",
#         eval_steps=cfg.training.eval_steps if eval_dataset else None,
#         save_strategy="steps",
#         save_steps=cfg.training.save_steps,
#         # save_total_limit=cfg.training.save_total_limit,
#         load_best_model_at_end=True if eval_dataset else False,
#         metric_for_best_model="eval_loss" if eval_dataset else None,
#         greater_is_better=False,
#         bf16=cfg.training.bf16,
#         fp16=cfg.training.fp16,
#         dataloader_pin_memory=False,
#         remove_unused_columns=False,
#         report_to="none",  # Disable wandb for this simple script
#         run_name=f"lora-{cfg.model.base_model.split('/')[-1]}-{cfg.experiment_name}",
#     )
    
#     # Create trainer
#     trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         tokenizer=tokenizer,
#         # formatting_func=formatting_prompts_func,
#         max_seq_length=cfg.training.max_seq_length,
#         packing=cfg.training.packing,
#         # Add tokenization parameters
#         dataset_text_field="text",  # if your data has a 'text' field
#         # dataset_kwargs={
#         #     "add_special_tokens": False,
#         #     "padding": True,
#         #     "truncation": True,
#         # }
#     )
    
#     # Start training
#     print("Starting LoRA fine-tuning...")
#     trainer.train()
    
#     # Save the final model
#     print(f"Saving LoRA adapter to: {cfg.paths.lora_output_dir}")
#     trainer.save_model()
#     tokenizer.save_pretrained(cfg.paths.lora_output_dir)
    
#     # Save training info
#     with open(os.path.join(cfg.paths.lora_output_dir, "training_info.txt"), "w") as f:
#         f.write(f"Base model: {cfg.model.base_model}\n")
#         f.write(f"LoRA rank: {cfg.lora.rank}\n")
#         f.write(f"LoRA alpha: {cfg.lora.alpha}\n")
#         f.write(f"Target modules: {cfg.lora.target_modules}\n")
#         f.write(f"Training samples: {len(train_dataset)}\n")
#         f.write(f"Epochs: {cfg.training.num_train_epochs}\n")
#         f.write(f"Learning rate: {cfg.training.learning_rate}\n")
    
#     print("LoRA fine-tuning completed!")
#     print(f"Adapter saved in: {cfg.paths.lora_output_dir}")


# if __name__ == "__main__":
#     main()


