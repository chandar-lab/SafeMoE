import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from mergoo.compose_experts import ComposeExperts
import datasets
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
from transformers import TrainingArguments, TrainerCallback, AutoTokenizer, AutoModelForCausalLM
from mergoo.models.modeling_mistral import MistralForCausalLM
from accelerate import Accelerator
import traceback
import pdb


def unified_formatting_func(example, mode="sft"):

    output_texts = []

    # Determine the question key
    question_key = "prompt" if "prompt" in example else "instruction"

    for i in range(len(example[question_key])):
        if mode == "sft" and "response" in example:
            # SFT mode: include both question and answer
            text = f"### Question: {example[question_key][i]}\n### Answer: {example['response'][i]}"
        else:
            # GRPO mode: only include question with answer prompt
            text = f"### Question: {example[question_key][i]}\n### Answer:"

        output_texts.append(text)

    return output_texts




SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def make_conversation(example):
    question_key = "prompt" if "prompt" in example else "instruction"
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example[question_key]},
        ],
    }

from math_verify import LatexExtractionConfig, parse, verify
def accuracy_reward(completions, **kwargs):
    return [1.0 ]
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['response']
    # question_key = "prompt" if "prompt" in completions[0] else "instruction"
    completion_contents = [completion[0]['prompt'] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards



import re
def format_reward(completions, **kwargs):
    return [1.0 ]
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["prompt"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]


@hydra.main(version_base=None, config_path="conf", config_name="config_grpo")
def main(cfg: DictConfig) -> None:
    # Set global config for reward function access
    global _global_config
    _global_config = cfg

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Check training mode
    grpo_mode = cfg.get('grpo', {}).get('mode', 'mixed')
    # use_grpo = cfg.get('grpo', {}).get('enabled', False)
    use_grpo = True if grpo_mode=="GRPO" or grpo_mode=="mixed" else False

    print(f"Training Mode: {grpo_mode}")
    print(f"GRPO Training: {'Enabled' if use_grpo else 'Disabled'}")

    # Check Llama Guard configuration
    llama_guard_enabled = cfg.get('llama_guard', {}).get('enabled', True)
    llama_guard_model = cfg.get('llama_guard', {}).get('model_id', 'meta-llama/Llama-Guard-3-8B')
    print(f"Llama Guard: {'Enabled' if llama_guard_enabled else 'Disabled'} ({llama_guard_model})")

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

    # Load model and tokenizer
    accelerator.print("Loading model and tokenizer...")

    # Load tokenizer first and configure pad token
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    accelerator.print(f"Tokenizer configured - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

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
        'train': train_testvalid['train'], #.select(range(100))
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

    # Handle different training modes
    if grpo_mode == "mixed":
        

        # Phase 2: GRPO on whole dataset
        print("Phase 2: GRPO training on whole dataset...")

        # Prepare dataset for GRPO (whole dataset)
        def prepare_grpo_dataset(examples):
            prompts = unified_formatting_func(examples, mode="grpo")
            return {"prompt": prompts}

        grpo_dataset = dataset.map(prepare_grpo_dataset, batched=True, remove_columns=dataset['train'].column_names)
        grpo_dataset = grpo_dataset['train'].select(range(min(50, len(grpo_dataset['train']))))
        # grpo_dataset = dataset.map(make_conversation)['train'].select(range(min(50, len(dataset['train']))))
        # test_dataset = test_dataset.map(make_conversation)
        
        
        # Set tokenizer padding for GRPO
        tokenizer.padding_side = "left"

        # GRPO Configuration
        # Configure training arguments using GRPOConfig
        grpo_config = GRPOConfig(
            output_dir=f"{cfg.paths.output_dir}/grpo_phase",
            learning_rate=cfg.grpo.get('learning_rate', cfg.training.learning_rate),
            
            remove_unused_columns=False, # to access the solution column in accuracy_reward
            
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            num_train_epochs=cfg.grpo.get('num_train_epochs', cfg.training.num_train_epochs),
            bf16=cfg.training.bf16,

            # Parameters that control de data preprocessing
            max_completion_length=cfg.grpo.get('max_completion_length', 64), # default: 256
            num_generations=cfg.grpo.get('num_generations', 4), # default: 8
            max_prompt_length=cfg.grpo.get('max_prompt_length', 128), # default: 512

            # Parameters related to reporting and saving
            # report_to=["tensorboard"],
            logging_steps=cfg.training.logging_steps,
            # push_to_hub=True,
            save_strategy=cfg.training.save_strategy,
            save_steps=cfg.training.save_steps,
            # wandb configuration
            report_to="wandb" if cfg.wandb.enabled and accelerator.is_main_process else "none",
            run_name=f"{cfg.wandb.run_name or cfg.experiment_name}_grpo",
        )


        
        # grpo_config = GRPOConfig(
        #     # output_dir=f"{cfg.paths.output_dir}/grpo_phase",
        #     per_device_train_batch_size=cfg.grpo.get('per_device_train_batch_size', cfg.training.per_device_train_batch_size),
        #     per_device_eval_batch_size=cfg.grpo.get('per_device_eval_batch_size', cfg.training.per_device_eval_batch_size),
        #     # learning_rate=cfg.grpo.get('learning_rate', cfg.training.learning_rate),
        #     save_total_limit=cfg.training.save_total_limit,
        #     # num_train_epochs=cfg.grpo.get('num_train_epochs', cfg.training.num_train_epochs),
        #     logging_strategy=cfg.training.logging_strategy,
        #     logging_steps=cfg.training.logging_steps,
        #     # gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        #     # bf16=cfg.training.bf16,
        #     logging_first_step=cfg.training.logging_first_step,
        #     save_strategy=cfg.training.save_strategy,
        #     save_steps=cfg.training.save_steps,
        #     # GRPO specific parameters
        #     # num_generations=cfg.grpo.get('num_generations', 4),
        #     # max_completion_length=cfg.grpo.get('max_completion_length', 256),
        #     # max_prompt_length=cfg.grpo.get('max_prompt_length', 512),
        #     temperature=cfg.grpo.get('temperature', 0.7),
        #     beta=cfg.grpo.get('beta', 0.0),  # KL coefficient
        #     # Generation parameters for stability
        #     # do_sample=True,
        #     top_p=0.9,
        #     top_k=50,
        #     min_p=0.01,  # Minimum probability threshold
        #     # wandb configuration
        #     # report_to="wandb" if cfg.wandb.enabled and accelerator.is_main_process else "none",
        #     # run_name=f"{cfg.wandb.run_name or cfg.experiment_name}_grpo",
        # )


        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward, accuracy_reward],
            args=grpo_config,
            train_dataset=grpo_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # # Create GRPO trainer
        # grpo_trainer = GRPOTrainer(
        #     # model=model,
        #     # args=grpo_config,
        #     # train_dataset=grpo_dataset, #grpo_dataset['train'],
        #     processing_class=tokenizer,
        #     eval_dataset=grpo_dataset, #grpo_dataset['val'],
        #     reward_funcs=[format_reward, accuracy_reward],
        #     callbacks=[WandbCallback()] if cfg.wandb.enabled else []
        # )

        # # Train GRPO phase
        # print("Starting GRPO training...")
        # grpo_trainer.train()
        # print("GRPO training completed!")

        # # except Exception as e:
        # #     print(f"Error in GRPO training: {e}")
        # #     traceback.print_exc()
        # #     pdb.post_mortem()



        # # Set final trainer for logging purposes
        # trainer = grpo_trainer

    else:
        raise ValueError(f"Invalid GRPO mode: {grpo_mode}")

    # Print training configuration
    if grpo_mode == "mixed":
        print(f"Training mode: Mixed (SFT + GRPO)")
        print("SFT phase completed, GRPO phase completed")
    elif grpo_mode == "GRPO":
        print(f"Training mode: GRPO")
        print(f"Number of generations per prompt: {trainer.args.num_generations}")
        print(f"Max completion length: {trainer.args.max_completion_length}")
        print(f"Temperature: {trainer.args.temperature}")
        print(f"Beta (KL coefficient): {trainer.args.beta}")
    else:
        print(f"Training mode: SFT")
        # print(f"Evaluation strategy: {trainer_args.evaluation_strategy}")
        # print(f"Eval steps: {trainer_args.eval_steps}")
        # print(f"Metric for best model: {trainer_args.metric_for_best_model}")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    accelerator.print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # Log model architecture info to wandb (only on main process)
    if cfg.wandb.enabled and accelerator.is_main_process:
        log_data = {
            "model_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "n_weights": n_weights,
            "n_router_weights": n_router_weights,
            "dataset_train_size": len(dataset['train']),
            "dataset_val_size": len(dataset['val']),
            "dataset_test_size": len(dataset['test']),
            "training_mode": "GRPO" if use_grpo else "SFT",
        }

        if grpo_mode == "GRPO" or grpo_mode == "mixed":
            log_data.update({
                "grpo_num_generations": trainer.args.num_generations,
                "grpo_max_completion_length": trainer.args.max_completion_length,
                "grpo_temperature": trainer.args.temperature,
                "grpo_beta": trainer.args.beta,
            })

        accelerator.log(log_data)

    # Training execution based on mode
    if grpo_mode == "mixed":
        # Mixed mode training was already completed in the trainer setup
        accelerator.print("Mixed training (SFT + GRPO) completed!")
    else:
        # Start training for single-mode training
        accelerator.print(f"Starting {grpo_mode} training...")
        trainer.train()
        accelerator.print(f"{grpo_mode} training completed!")

    # Run final evaluation (only for SFT, GRPO handles evaluation differently)
    if grpo_mode == "SFT":
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




























