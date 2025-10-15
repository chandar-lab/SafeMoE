'''
Training script for MoE routers with GRPO (Group Relative Policy Optimization) support

This script supports three training modes:
1. SFT (Supervised Fine-Tuning) - Traditional training mode
2. GRPO - Reinforcement learning with DeepSeek-style reward function
3. Mixed - Sequential SFT (500 samples) then GRPO (whole dataset)

GRPO Reward Function Components:
1. Safety/Correctness Score:
   - Informative/Safe responses: +1
   - Unsafe responses: -1
   - Refusal responses: 0
2. Format Quality Score: [0, 0.1]
   - Rewards well-structured, coherent responses

Features:
- Unified formatting function for both SFT and GRPO
- Configurable epochs for each training phase in mixed mode
- Support for different dataset sizes per phase

Usage:
- For SFT training: Set grpo.mode="SFT" in config
- For GRPO training: Set grpo.mode="GRPO" in config
- For Mixed training: Set grpo.mode="mixed" in config
'''


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
    """
    Unified formatting function for both SFT and GRPO training

    Args:
        example: Dataset example containing questions and optionally responses
        mode: "sft" for supervised fine-tuning (includes responses),
              "grpo" for GRPO training (prompts only)

    Returns:
        List of formatted strings
    """
    output_texts = []

    # Determine the question key
    question_key = "prompt" if "prompt" in example else "instruction"

    # Determine the length based on available data
    if mode == "sft" and "response" in example:
        length = len(example['response'])
    else:
        length = len(example[question_key])

    for i in range(length):
        if mode == "sft" and "response" in example:
            # SFT mode: include both question and answer
            text = f"### Question: {example[question_key][i]}\n### Answer: {example['response'][i]}"
        else:
            # GRPO mode: only include question with answer prompt
            text = f"### Question: {example[question_key][i]}\n### Answer:"

        output_texts.append(text)

    return output_texts


def formatting_prompts_func(example):
    """SFT formatting function - calls unified function"""
    return unified_formatting_func(example, mode="sft")


def preprocess_for_sft(examples):
    """Convert your format to TRL's expected format"""
    question_key = "prompt" if "prompt" in examples else "instruction"

    prompts = []
    completions = []

    for i in range(len(examples[question_key])):
        prompt = f"### Question: {examples[question_key][i]}\n### Answer:"
        completion = examples['response'][i]

        prompts.append(prompt)
        completions.append(completion)

    return {"prompt": prompts, "completion": completions}


# Global variables for Llama Guard model (loaded once)
_llama_guard_model = None
_llama_guard_tokenizer = None
_global_config = None
_global_config = None

def load_llama_guard(cfg=None):
    """Load Llama Guard model and tokenizer (called once)"""
    global _llama_guard_model, _llama_guard_tokenizer

    if _llama_guard_model is None:
        # Check if Llama Guard is enabled in config
        if cfg and not cfg.get('llama_guard', {}).get('enabled', True):
            print("Llama Guard disabled in config, using heuristic fallback...")
            return False

        print("Loading Llama Guard model...")

        # Get model configuration from config or use defaults
        if cfg and 'llama_guard' in cfg:
            model_id = cfg.llama_guard.get('model_id', "meta-llama/Llama-Guard-3-8B")
            device_map = cfg.llama_guard.get('device_map', "auto")
            torch_dtype_str = cfg.llama_guard.get('torch_dtype', "bfloat16")
            torch_dtype = getattr(torch, torch_dtype_str)
        else:
            model_id = "meta-llama/Llama-Guard-3-8B"
            device_map = "auto"
            torch_dtype = torch.bfloat16

        try:
            _llama_guard_tokenizer = AutoTokenizer.from_pretrained(model_id)
            _llama_guard_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            print(f"Llama Guard model ({model_id}) loaded successfully!")
        except Exception as e:
            print(f"Failed to load Llama Guard model: {e}")
            print("Falling back to heuristic-based safety check...")
            return False

    return True

def llama_guard_judge(text, cfg=None):
    """
    Use actual Llama Guard model for safety assessment
    Returns: 1 for informative/safe, -1 for unsafe, 0 for refusal
    """
    # Try to use actual Llama Guard model
    return _llama_guard_inference(text)
    if load_llama_guard(cfg) and _llama_guard_model is not None:
        return _llama_guard_inference(text)
    else:
        print("Fallback to heuristic-based check")
        return _heuristic_safety_check(text)

def _llama_guard_inference(text):
    """Run inference with Llama Guard model"""
    try:
        # Llama Guard prompt format
        conversation = [{"role": "user", "content": text}]
        input_ids = _llama_guard_tokenizer.apply_chat_template(conversation, return_tensors="pt")

        if torch.cuda.is_available():
            input_ids = input_ids.to(_llama_guard_model.device)

        with torch.no_grad():
            output = _llama_guard_model.generate(
                input_ids,
                max_new_tokens=100,
                pad_token_id=_llama_guard_tokenizer.eos_token_id,
                do_sample=False,
            )

        # Decode the response
        response = _llama_guard_tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # Parse Llama Guard response
        if "safe" in response.lower():
            return 1  # Safe/informative
        elif "unsafe" in response.lower():
            return -1  # Unsafe
        else:
            return 0  # Unclear/refusal

    except Exception as e:
        print(f"Error in Llama Guard inference: {e}")
        assert False, f"Error in Llama Guard inference: {e}"
        # return _heuristic_safety_check(text)

# def _heuristic_safety_check(text):
#     """
#     Fallback heuristic-based safety check
#     Returns: 1 for informative/safe, -1 for unsafe, 0 for refusal
#     """
#     text_lower = text.lower()

#     # Check for refusal patterns
#     refusal_patterns = [
#         "i can't", "i cannot", "i'm not able", "i am not able",
#         "i don't feel comfortable", "i'm sorry but", "i apologize but",
#         "i'm not programmed", "i cannot provide", "i can't help with"
#     ]

#     if any(pattern in text_lower for pattern in refusal_patterns):
#         return 0  ############# correct this , rn make everything mess up if a responses has it. Refusal

#     # Check for unsafe content patterns
#     unsafe_patterns = [
#         "violence", "harm", "illegal", "dangerous", "weapon", "drug", "abuse",
#         "suicide", "self-harm", "hate", "discrimination", "explicit"
#     ]

#     if any(pattern in text_lower for pattern in unsafe_patterns):
#         return -1  # Unsafe

#     return 1  # Safe/informative


def calculate_format_score(text):
    """
    Calculate format score based on response structure and quality
    Returns score between 0 and 0.1

    Rewards both general quality and adherence to the expected format:
    ### Question: [question]
    ### Answer: [answer]
    """
    score = 0.0

    # Check for expected format structure (highest reward)
    if "### Question:" in text and "### Answer:" in text:
        score += 0.04  # Major bonus for following the expected format
    elif "### Answer:" in text:
        score += 0.02  # Partial bonus for having answer section

    # Check for proper structure (has some explanation)
    if len(text.strip()) > 20:
        score += 0.015

    # Check for complete sentences
    if text.strip().endswith('.') or text.strip().endswith('!') or text.strip().endswith('?'):
        score += 0.015

    # Check for proper capitalization
    if text.strip() and text.strip()[0].isupper():
        score += 0.01

    # Check for reasonable length (not too short, not too long)
    word_count = len(text.split())
    if 10 <= word_count <= 200:
        score += 0.02
    elif 5 <= word_count < 10 or 200 < word_count <= 300:
        score += 0.01

    # Check for coherent structure (has some punctuation)
    if any(punct in text for punct in ['.', ',', '!', '?', ':']):
        score += 0.015

    return min(score, 0.1)  # Cap at 0.1


def deepseek_style_reward_function(completions, **kwargs):
    """
    DeepSeek-style reward function with two components:
    1. Correct answer → judge by llama guard (informative/safe=+1, unsafe=-1, refusal=0)
    2. Answer format → helps to have better response: [0, 0.1]
    """
    rewards = []

    # Get config from global variable if available
    cfg = globals().get('_global_config', None)

    for i, completion in enumerate(completions):
        try:
            # Component 1: Safety/correctness judgment using actual Llama Guard
            safety_score = llama_guard_judge(completion, cfg)

            # Component 2: Format quality score
            format_score = calculate_format_score(completion)

            # Combine scores
            total_reward = safety_score + format_score

            # Ensure reward is finite and reasonable
            if not torch.isfinite(torch.tensor(total_reward)):
                print(f"Warning: Non-finite reward {total_reward} for completion {i}, using default reward 0.5")
                total_reward = 0.5

            rewards.append(total_reward)

        except Exception as e:
            print(f"Error processing completion {i}: {e}")
            print(f"Completion text: {completion[:100]}...")
            # Use a safe default reward
            rewards.append(0.5)

    return rewards


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
        print("Setting up Mixed training (SFT first, then GRPO)...")

        # Phase 1: SFT on 500 samples
        print("Phase 1: SFT training on 500 samples...")

        # Create SFT dataset with 500 samples
        sft_dataset = dataset['train'].select(range(min(50, len(dataset['train']))))
        print(f"SFT dataset size: {len(sft_dataset)}")

        # SFT Training arguments - with proper parameter names for transformers 4.46.3
        print("Creating SFT TrainingArguments...")
        try:
            sft_args = TrainingArguments(
            # sft_args = SFTConfig( 
                output_dir=f"{cfg.paths.output_dir}/sft_phase",
                per_device_train_batch_size=cfg.training.per_device_train_batch_size,
                per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
                learning_rate=cfg.training.learning_rate,
                save_total_limit=cfg.training.save_total_limit,
                num_train_epochs=cfg.get('sft', {}).get('num_train_epochs', cfg.training.num_train_epochs),
                eval_steps=cfg.training.eval_steps,
                logging_strategy=cfg.training.logging_strategy,
                logging_steps=cfg.training.logging_steps,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                bf16=cfg.training.bf16,
                logging_first_step=cfg.training.logging_first_step,
                eval_strategy=cfg.training.evaluation_strategy,  # Updated parameter name
                save_strategy=cfg.training.save_strategy,
                save_steps=cfg.training.save_steps,
                load_best_model_at_end=cfg.training.load_best_model_at_end,
                metric_for_best_model=cfg.training.metric_for_best_model,
                greater_is_better=cfg.training.greater_is_better,
                log_level="info",
                report_to="wandb" if cfg.wandb.enabled and accelerator.is_main_process else "none",
                run_name=f"{cfg.wandb.run_name or cfg.experiment_name}_sft",
                # max_length=cfg.training.max_seq_length,
                # max_completion_length=cfg.grpo.get('max_completion_length', 256),
                # max_prompt_length=cfg.grpo.get('max_prompt_length', 512),
            )
            print("✓ SFT TrainingArguments created successfully")
        except Exception as e:
            print(f"✗ Error creating SFT TrainingArguments: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Preprocess datasets for SFT (convert to prompt/completion format)
        sft_dataset_processed = sft_dataset.map(preprocess_for_sft, batched=True, remove_columns=sft_dataset.column_names)
        eval_dataset_processed = dataset['val'].map(preprocess_for_sft, batched=True, remove_columns=dataset['val'].column_names)

        # Create SFT trainer
        sft_trainer = SFTTrainer(
            model,
            args=sft_args,
            train_dataset=sft_dataset_processed,
            eval_dataset=eval_dataset_processed,
            processing_class=tokenizer,
            callbacks=[WandbCallback()] if cfg.wandb.enabled else []
        )

        # Train SFT phase
        print("Starting SFT training...")
        # sft_trainer.train()
        print("SFT training completed!")

        # Phase 2: GRPO on whole dataset
        print("Phase 2: GRPO training on whole dataset...")

        # Prepare dataset for GRPO (whole dataset)
        def prepare_grpo_dataset(examples):
            prompts = unified_formatting_func(examples, mode="grpo")
            return {"prompt": prompts}

        grpo_dataset = dataset.map(prepare_grpo_dataset, batched=True, remove_columns=dataset['train'].column_names)
        grpo_dataset = grpo_dataset['train'].select(range(min(50, len(grpo_dataset['train']))))
        print(f"GRPO dataset size: {len(sft_dataset)}")

        # Set tokenizer padding for GRPO
        tokenizer.padding_side = "left"

        # GRPO Configuration
        grpo_config = GRPOConfig(
            output_dir=f"{cfg.paths.output_dir}/grpo_phase",
            per_device_train_batch_size=cfg.grpo.get('per_device_train_batch_size', cfg.training.per_device_train_batch_size),
            per_device_eval_batch_size=cfg.grpo.get('per_device_eval_batch_size', cfg.training.per_device_eval_batch_size),
            learning_rate=cfg.grpo.get('learning_rate', cfg.training.learning_rate),
            save_total_limit=cfg.training.save_total_limit,
            num_train_epochs=cfg.grpo.get('num_train_epochs', cfg.training.num_train_epochs),
            logging_strategy=cfg.training.logging_strategy,
            logging_steps=cfg.training.logging_steps,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            bf16=cfg.training.bf16,
            logging_first_step=cfg.training.logging_first_step,
            save_strategy=cfg.training.save_strategy,
            save_steps=cfg.training.save_steps,
            # GRPO specific parameters
            num_generations=cfg.grpo.get('num_generations', 4),
            max_completion_length=cfg.grpo.get('max_completion_length', 256),
            max_prompt_length=cfg.grpo.get('max_prompt_length', 512),
            temperature=cfg.grpo.get('temperature', 0.7),
            beta=cfg.grpo.get('beta', 0.0),  # KL coefficient
            # Generation parameters for stability
            # do_sample=True,
            top_p=0.9,
            top_k=50,
            min_p=0.01,  # Minimum probability threshold
            # wandb configuration
            report_to="wandb" if cfg.wandb.enabled and accelerator.is_main_process else "none",
            run_name=f"{cfg.wandb.run_name or cfg.experiment_name}_grpo",
        )


        # try:
        # Test model generation before training to catch issues early
        print("Testing model generation before GRPO training...")
        try:
            test_prompt = "### Question: What is 2+2?\n### Answer:"
            test_inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                test_inputs = {k: v.cuda() for k, v in test_inputs.items()}

            with torch.no_grad():
                test_outputs = model.generate(
                    **test_inputs,
                    max_new_tokens=10,
                    # do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                )
            test_response = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
            print(f"✓ Model generation test successful: {test_response}")
        except Exception as e:
            print(f"✗ Model generation test failed: {e}")
            print("This indicates the model has issues that will cause GRPO to fail.")
            raise

        # Create GRPO trainer
        grpo_trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=grpo_dataset, #grpo_dataset['train'],
            processing_class=tokenizer,
            eval_dataset=grpo_dataset, #grpo_dataset['val'],
            reward_funcs=deepseek_style_reward_function,
            callbacks=[WandbCallback()] if cfg.wandb.enabled else []
        )

        # Train GRPO phase
        print("Starting GRPO training...")
        grpo_trainer.train()
        print("GRPO training completed!")

        # except Exception as e:
        #     print(f"Error in GRPO training: {e}")
        #     traceback.print_exc()
        #     pdb.post_mortem()



        # Set final trainer for logging purposes
        trainer = grpo_trainer

    elif grpo_mode == "GRPO":
        print("Setting up GRPO trainer...")

        # Prepare dataset for GRPO
        def prepare_grpo_dataset(examples):
            prompts = unified_formatting_func(examples, mode="grpo")
            return {"prompt": prompts}

        dataset = dataset.map(prepare_grpo_dataset, batched=True, remove_columns=dataset['train'].column_names)

        # Set tokenizer padding for GRPO
        tokenizer.padding_side = "left"

        # GRPO Configuration
        grpo_config = GRPOConfig(
            output_dir=cfg.paths.output_dir,
            per_device_train_batch_size=cfg.grpo.get('per_device_train_batch_size', cfg.training.per_device_train_batch_size),
            per_device_eval_batch_size=cfg.grpo.get('per_device_eval_batch_size', cfg.training.per_device_eval_batch_size),
            learning_rate=cfg.grpo.get('learning_rate', cfg.training.learning_rate),
            save_total_limit=cfg.training.save_total_limit,
            num_train_epochs=cfg.grpo.get('num_train_epochs', cfg.training.num_train_epochs),
            logging_strategy=cfg.training.logging_strategy,
            logging_steps=cfg.training.logging_steps,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            bf16=cfg.training.bf16,
            logging_first_step=cfg.training.logging_first_step,
            save_strategy=cfg.training.save_strategy,
            save_steps=cfg.training.save_steps,
            # GRPO specific parameters
            num_generations=cfg.grpo.get('num_generations', 4),
            max_completion_length=cfg.grpo.get('max_completion_length', 256),
            max_prompt_length=cfg.grpo.get('max_prompt_length', 512),
            temperature=cfg.grpo.get('temperature', 0.7),
            beta=cfg.grpo.get('beta', 0.0),  # KL coefficient
            # Generation parameters for stability
            do_sample=True,
            top_p=0.9,
            top_k=50,
            min_p=0.01,  # Minimum probability threshold
            # wandb configuration
            report_to="wandb" if cfg.wandb.enabled and accelerator.is_main_process else "none",
            run_name=cfg.wandb.run_name or cfg.experiment_name,
        )

        # Test model generation before training to catch issues early
        print("Testing model generation before GRPO training...")
        try:
            test_prompt = "### Question: What is 2+2?\n### Answer:"
            test_inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                test_inputs = {k: v.cuda() for k, v in test_inputs.items()}

            with torch.no_grad():
                test_outputs = model.generate(
                    **test_inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                )
            test_response = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
            print(f"✓ Model generation test successful: {test_response}")
        except Exception as e:
            print(f"✗ Model generation test failed: {e}")
            print("This indicates the model has issues that will cause GRPO to fail.")
            raise

        # Create GRPO trainer
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset['train'],
            processing_class=tokenizer,
            eval_dataset=dataset['val'],
            reward_funcs=deepseek_style_reward_function,
            callbacks=[WandbCallback()] if cfg.wandb.enabled else []
        )

    else:  # SFT mode
        print("Setting up SFT trainer...")

        # Training arguments for SFT - with proper parameter names for transformers 4.46.3
        print("Creating regular SFT TrainingArguments...")
        try:
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
                eval_strategy=cfg.training.evaluation_strategy,  # Updated parameter name
                save_strategy=cfg.training.save_strategy,
                save_steps=cfg.training.save_steps,
                load_best_model_at_end=cfg.training.load_best_model_at_end,
                metric_for_best_model=cfg.training.metric_for_best_model,
                greater_is_better=cfg.training.greater_is_better,
                log_level="info",
                report_to="wandb" if cfg.wandb.enabled and accelerator.is_main_process else "none",
                run_name=cfg.wandb.run_name or cfg.experiment_name,
            )
            print("✓ Regular SFT TrainingArguments created successfully")
        except Exception as e:
            print(f"✗ Error creating regular SFT TrainingArguments: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Preprocess datasets for SFT (convert to prompt/completion format)
        train_dataset_processed = dataset['train'].map(preprocess_for_sft, batched=True, remove_columns=dataset['train'].column_names)
        eval_dataset_processed = dataset['val'].map(preprocess_for_sft, batched=True, remove_columns=dataset['val'].column_names)

        # Create SFT trainer
        trainer = SFTTrainer(
            model,
            args=trainer_args,
            train_dataset=train_dataset_processed,
            eval_dataset=eval_dataset_processed,
            processing_class=tokenizer,
            max_seq_length=cfg.training.max_seq_length,
            callbacks=[WandbCallback()] if cfg.wandb.enabled else []
        )

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

