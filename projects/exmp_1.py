'''
Correct one for traning the routers
'''


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from mergoo.compose_experts import ComposeExperts
import datasets
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer, TrainerCallback
from mergoo.models.modeling_mistral import MistralForCausalLM
from accelerate import Accelerator
import os


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['response'])):
        question_key = "prompt" if "prompt" in example else "instruction"
        # return [f"### Question: {q}\n### Answer: {a}" for q, a in zip(example[question_key], example["response"])]
        text = f"### Question: {example[question_key][i]}\n ### Answer: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize Accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision="bf16" if cfg.training.bf16 else "no",
        log_with="wandb" if cfg.wandb.enabled else None,
        project_dir=cfg.paths.output_dir,
    )

    # Initialize wandb with Hydra config (only on main process)
    if accelerator.is_main_process and cfg.wandb.enabled:
        accelerator.init_trackers(
            project_name=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={
                "wandb": {
                    "name": cfg.wandb.run_name or cfg.experiment_name,
                    "tags": cfg.wandb.get("tags", []),
                    "notes": cfg.wandb.get("notes", ""),
                }
            }
        )

    # Create model configuration from hydra config
    model_config = {
        "model_type": cfg.model.model_type,
        "num_experts_per_tok": cfg.model.num_experts_per_tok,
        "base_model": cfg.model.base_model,
        "experts": OmegaConf.to_container(cfg.model.experts, resolve=True),
    }

    # Create checkpoint
    print("Creating model checkpoint...")
    expertmerger = ComposeExperts(model_config, torch_dtype=torch.float16)
    expertmerger.compose()
    expertmerger.save_checkpoint(cfg.paths.model_checkpoint)

    # Load model
    accelerator.print("Loading model...")

    # For DeepSpeed, we don't use device_map="auto"
    # if cfg.deepspeed.enabled:
    #     model = MistralForCausalLM.from_pretrained(
    #         cfg.paths.model_checkpoint,
    #         torch_dtype=torch.bfloat16,
    #     )
    # else:
    #     model = MistralForCausalLM.from_pretrained(
    #         cfg.paths.model_checkpoint,
    #         device_map="auto" if not accelerator.distributed_type else None,
    #         torch_dtype=torch.bfloat16,
    #     )

    model = MistralForCausalLM.from_pretrained(
            cfg.paths.model_checkpoint,
            torch_dtype=torch.bfloat16,
        )

    # Train only router (gating) layers
    n_weights, n_router_weights = 0, 0
    for name, weight in model.named_parameters():
        if "gate" in name:
            weight.requires_grad_(True)
            n_router_weights += 1
        else:
            weight.requires_grad_(False)
        n_weights += 1
    print(f"Total weights: {n_weights}, Router weights: {n_router_weights}")

    # Load and prepare dataset
    print("Loading dataset...")
    # dataset = datasets.load_dataset(cfg.data.dataset_name)['train']
    dataset = datasets.load_dataset("json", data_files=cfg.data.dataset_name)["train"]

    # Split dataset
    train_testvalid = dataset.train_test_split(test_size=cfg.data.test_size)
    test_valid = train_testvalid['test'].train_test_split(test_size=cfg.data.test_valid_split)

    dataset = datasets.DatasetDict({
        'train': train_testvalid['train'].select(range(min(2000, len(train_testvalid['train'])))), #.select(range(100))
        'test': test_valid['test'],
        'val': test_valid['train'],
    })

    print(f"Dataset sizes - Train: {len(dataset['train'])}, Val: {len(dataset['val'])}, Test: {len(dataset['test'])}")

    # Custom wandb callback for additional logging
    class WandbCallback(TrainerCallback):
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if logs:
                # Log router/gate specific metrics if available
                router_logs = {k: v for k, v in logs.items() if 'gate' in k.lower() or 'router' in k.lower()}
                if router_logs:
                    # wandb.log(router_logs, step=state.global_step)
                    accelerator.log(router_logs, step=state.global_step)

                # Explicitly log evaluation loss if available
                eval_logs = {k: v for k, v in logs.items() if k.startswith('eval_')}
                if eval_logs:
                    print(f"Evaluation metrics at step {state.global_step}: {eval_logs}")
                    if cfg.wandb.enabled:
                        # wandb.log(eval_logs, step=state.global_step)
                        accelerator.log(eval_logs, step=state.global_step)

    # Training arguments
    trainer_args = TrainingArguments(
        output_dir=cfg.paths.output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_train_epochs,
        eval_steps=cfg.training.eval_steps,
        logging_strategy=cfg.training.logging_strategy,
        logging_steps=cfg.training.logging_steps,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        bf16=cfg.training.bf16,
        logging_first_step=cfg.training.logging_first_step,
        eval_strategy=cfg.training.evaluation_strategy,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        # Ensure evaluation loss is logged
        include_inputs_for_metrics=True,
        log_level="info",
        # wandb configuration
        report_to="wandb" if cfg.wandb.enabled and accelerator.is_main_process else "none",
        run_name=cfg.wandb.run_name or cfg.experiment_name,
        # DeepSpeed configuration
        # deepspeed=cfg.deepspeed.config_path if cfg.deepspeed.enabled else None,
        # Distributed training settings
        # ddp_find_unused_parameters=False,
        # dataloader_pin_memory=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model,
        args=trainer_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        formatting_func=formatting_prompts_func,
        max_seq_length=cfg.training.max_seq_length,
        callbacks=[WandbCallback()] if cfg.wandb.enabled else []
    )

    # Ensure evaluation is properly configured
    print(f"Evaluation strategy: {trainer_args.evaluation_strategy}")
    print(f"Eval steps: {trainer_args.eval_steps}")
    print(f"Metric for best model: {trainer_args.metric_for_best_model}")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    accelerator.print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # Log model architecture info to wandb (only on main process)
    if cfg.wandb.enabled and accelerator.is_main_process:
        accelerator.log({
            "model_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "n_weights": n_weights,
            "n_router_weights": n_router_weights,
            "dataset_train_size": len(dataset['train']),
            "dataset_val_size": len(dataset['val']),
            "dataset_test_size": len(dataset['test']),
        })

    # Start training
    accelerator.print("Starting training...")
    trainer.train()

    accelerator.print("Training completed!")

    # Run final evaluation to get final eval loss
    accelerator.print("Running final evaluation...")
    eval_results = trainer.evaluate()
    accelerator.print(f"Final evaluation results: {eval_results}")

    # Log final evaluation results
    if cfg.wandb.enabled and accelerator.is_main_process:
        final_eval_logs = {f"final_{k}": v for k, v in eval_results.items()}
        accelerator.log(final_eval_logs)

    # Finish tracking
    if cfg.wandb.enabled and accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()

