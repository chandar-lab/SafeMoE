"""
Train alpha coefficients for MoE models.

This script implements a new approach where for each layer of the MoE model,
when merging LoRA with the base model, we add a trainable coefficient α:
    final_weight = α * w_base + (1-α) * δw_moe

The α parameters are trained using a mixed dataset of:
1. Safe samples from the evaluation data (where "is_safe" is True)
2. 20% of the Alpaca dataset (tatsu-lab/alpaca)

Only the α parameters are trained; everything else is frozen.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable
import argparse
from tqdm import tqdm
import random
import logging
import functools
import inspect
from functools import partial
from torch.utils.checkpoint import checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_mistral import MistralForCausalLM
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset, DatasetDict, load_dataset
import gc


class AlphaMoELinear(nn.Module):
    """
    Enhanced MoE Linear layer with trainable alpha coefficients.
    Modifies the existing MoE layer to include: α * w_base + (1-α) * δw_moe
    """
    def __init__(self, original_moe_layer):
        super().__init__()
        self.original_layer = original_moe_layer

        # Copy all attributes from original layer
        for attr_name in dir(original_moe_layer):
            if not attr_name.startswith('_') and not callable(getattr(original_moe_layer, attr_name)):
                try:
                    setattr(self, attr_name, getattr(original_moe_layer, attr_name))
                except:
                    pass

        # Initialize alpha parameters for each expert
        self.alpha_coefficients = nn.ParameterDict()

        # Store base layer weights (frozen)
        if hasattr(original_moe_layer, 'base_layer'):
            self.register_buffer('base_weight', original_moe_layer.base_layer.weight.data.clone())

        # Initialize alpha for each adapter/expert
        for adapter_name in original_moe_layer.active_adapters:
            self.alpha_coefficients[adapter_name] = nn.Parameter(
                torch.tensor(0.5, dtype=torch.float32)
            )

    def forward(self, x, *args, **kwargs):
        """
        Modified forward pass that incorporates alpha coefficients
        """
        previous_dtype = x.dtype
        gate_logits = self.original_layer.gate(x)  # b,s,N
        gate_logits = F.softsign(gate_logits).to(previous_dtype)
        weights, selected_experts = torch.topk(
            gate_logits, self.original_layer.num_experts_per_tok
        )

        # Base layer output
        result = self.original_layer.base_layer(x, *args, **kwargs)

        # Apply alpha-weighted LoRA adapters
        for ix, active_adapter in enumerate(self.original_layer.active_adapters):
            if active_adapter not in self.original_layer.lora_A.keys():
                continue

            lora_A = self.original_layer.lora_A[active_adapter]
            lora_B = self.original_layer.lora_B[active_adapter]
            dropout = self.original_layer.lora_dropout[active_adapter]
            scaling = self.original_layer.scaling[active_adapter]

            # Get alpha coefficient for this adapter
            alpha = self.alpha_coefficients[active_adapter]

            x_typed = x.to(lora_A.weight.dtype)
            batch_idx, tok_idx, expert_idx = torch.where(selected_experts == ix)
            x_adapter = x_typed[batch_idx, tok_idx]

            # Compute LoRA delta: δw_moe = lora_B(lora_A(x)) * scaling
            delta_moe = lora_B(lora_A(dropout(x_adapter))) * scaling

            # Apply alpha weighting: α * base_output + (1-α) * δw_moe
            # Since result already contains base output, we modify the addition
            # The idea is: final = α * base + (1-α) * (base + delta) = base + (1-α) * delta
            weighted_delta = (1 - alpha) * delta_moe

            # Ensure weights are on the same device as weighted_delta
            expert_weights = weights[batch_idx, tok_idx, expert_idx].unsqueeze(-1)
            expert_weights = expert_weights.to(weighted_delta.device)

            result[batch_idx, tok_idx] += weighted_delta * expert_weights

        result = result.to(previous_dtype)
        return result


class AlphaMoEModel(nn.Module):
    """
    Model that replaces MoE layers with alpha-coefficient enhanced versions
    """
    def __init__(self, base_moe_model):
        super().__init__()
        self.base_model = base_moe_model
        self.alpha_layers = {}

        # Replace MoE layers with alpha-enhanced versions
        self._replace_moe_layers()

        # Freeze all parameters except alphas
        self._freeze_non_alpha_params()

    def _replace_moe_layers(self):
        """Replace MoE layers with alpha-enhanced versions"""
        def replace_layer(module, name_parts):
            for name, child in module.named_children():
                full_name = '.'.join(name_parts + [name])

                # Check if this is a MoE layer (has active_adapters and lora_A)
                if (hasattr(child, 'active_adapters') and
                    hasattr(child, 'lora_A') and
                    hasattr(child, 'base_layer')):

                    print(f"Replacing MoE layer: {full_name}")
                    alpha_layer = AlphaMoELinear(child)
                    setattr(module, name, alpha_layer)
                    self.alpha_layers[full_name] = alpha_layer
                else:
                    # Recursively check children
                    replace_layer(child, name_parts + [name])

        replace_layer(self.base_model, [])

    def _freeze_non_alpha_params(self):
        """Freeze all parameters except alpha coefficients"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Enable gradients for alpha parameters
        for layer_name, alpha_layer in self.alpha_layers.items():
            for alpha_param in alpha_layer.alpha_coefficients.values():
                alpha_param.requires_grad = True
                print(f"Enabled gradients for alpha in layer: {layer_name}")

    def forward(self, *args, **kwargs):
        """Forward pass through the model"""
        return self.base_model(*args, **kwargs)

    def get_alpha_parameters(self):
        """Get all alpha parameters for optimization"""
        alpha_params = []
        for alpha_layer in self.alpha_layers.values():
            alpha_params.extend(alpha_layer.alpha_coefficients.values())
        return alpha_params

    def print_alpha_values(self, title="Alpha values"):
        """Print current alpha values"""
        print(f"\n{title}:")
        for layer_name, alpha_layer in self.alpha_layers.items():
            for adapter_name, alpha in alpha_layer.alpha_coefficients.items():
                print(f"  {layer_name}.{adapter_name}: {alpha.item():.4f}")

    @property
    def supports_gradient_checkpointing(self):
        """Check if the base model supports gradient checkpointing"""
        return hasattr(self.base_model, 'supports_gradient_checkpointing') and self.base_model.supports_gradient_checkpointing

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.
        Delegates to the base model's gradient checkpointing implementation.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        # Delegate to base model
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            # Fallback implementation
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {"use_reentrant": True}

            gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

            # Check if using old format
            if hasattr(self.base_model, '_set_gradient_checkpointing'):
                _is_using_old_format = "value" in inspect.signature(self.base_model._set_gradient_checkpointing).parameters

                if not _is_using_old_format:
                    self.base_model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
                else:
                    self.base_model.apply(partial(self.base_model._set_gradient_checkpointing, value=True))

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        Delegates to the base model's gradient checkpointing implementation.
        """
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
        elif hasattr(self.base_model, '_set_gradient_checkpointing'):
            _is_using_old_format = "value" in inspect.signature(self.base_model._set_gradient_checkpointing).parameters

            if not _is_using_old_format:
                self.base_model._set_gradient_checkpointing(enable=False)
            else:
                self.base_model.apply(partial(self.base_model._set_gradient_checkpointing, value=False))

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        """
        if hasattr(self.base_model, 'is_gradient_checkpointing'):
            return self.base_model.is_gradient_checkpointing
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.base_model.modules())

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """
        if hasattr(self.base_model, 'enable_input_require_grads'):
            self.base_model.enable_input_require_grads()
        elif hasattr(self.base_model, 'get_input_embeddings'):
            def make_inputs_require_grads(module, input, output):
                output.requires_grad_(True)

            self._require_grads_hook = self.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """
        Removes the _require_grads_hook.
        """
        if hasattr(self.base_model, 'disable_input_require_grads'):
            self.base_model.disable_input_require_grads()
        elif hasattr(self, '_require_grads_hook'):
            self._require_grads_hook.remove()

    def get_input_embeddings(self):
        """Get input embeddings from the base model"""
        if hasattr(self.base_model, 'get_input_embeddings'):
            return self.base_model.get_input_embeddings()
        return None


def load_safe_samples(data_path: str) -> List[Dict]:
    """Load safe samples from the evaluation data"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Safe data file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    safe_samples = []
    detailed_results = data.get('detailed_results', [])

    if not detailed_results:
        raise ValueError("No 'detailed_results' found in the data file")

    for item in detailed_results:
        if item.get('is_safe', False):
            if 'instruction' in item and 'response' in item:
                safe_samples.append({
                    'instruction': item['instruction'],
                    'response': item['response']
                })

    if not safe_samples:
        raise ValueError("No safe samples found in the data file")

    logger.info(f"Loaded {len(safe_samples)} safe samples from evaluation data")
    return safe_samples


def load_alpaca_subset(percentage: float = 0.2) -> List[Dict]:
    """Load a subset of the Alpaca dataset"""
    try:
        logger.info(f"Loading {percentage*100}% of Alpaca dataset...")
        alpaca_dataset = load_dataset("tatsu-lab/alpaca")['train']

        # Sample the specified percentage
        sample_size = int(len(alpaca_dataset) * percentage)
        if sample_size == 0:
            logger.warning("Sample size is 0, using at least 1 sample")
            sample_size = 1

        indices = random.Random(42).sample(range(len(alpaca_dataset)), min(sample_size, len(alpaca_dataset)))

        alpaca_samples = []
        for idx in indices:
            item = alpaca_dataset[idx]
            # Combine instruction and input as specified
            instruction_input = item['instruction']
            if item.get('input', '').strip():
                instruction_input += f"\n{item['input']}"

            alpaca_samples.append({
                'instruction': instruction_input,
                'response': item['output']
            })

        logger.info(f"Loaded {len(alpaca_samples)} samples from Alpaca dataset ({percentage*100}%)")
        return alpaca_samples

    except Exception as e:
        logger.error(f"Error loading Alpaca dataset: {e}")
        logger.warning("Continuing without Alpaca data...")
        return []


def create_mixed_dataset(safe_data_path: str, alpaca_percentage: float = 0.2) -> DatasetDict:
    """Create mixed dataset from safe samples and Alpaca subset"""
    # Load safe samples
    data_size = 300
    safe_samples = load_safe_samples(safe_data_path)
    
    # Load Alpaca subset
    alpaca_samples = load_alpaca_subset(percentage=len(safe_samples))[:100]
    
    # Combine datasets
    all_samples = safe_samples + alpaca_samples
    random.Random(42).shuffle(all_samples)
    
    # Split into train/val
    split_idx = int(0.99 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Create Dataset objects
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'val': val_dataset
    })
    
    print(f"Created mixed dataset: {len(train_samples)} train, {len(val_samples)} val samples")
    return dataset_dict


def formatting_prompts_func(example):
    """Format prompts for training"""
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n### Answer: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the examples"""
    # Format the text
    texts = []
    for i in range(len(examples['instruction'])):
        text = f"### Question: {examples['instruction'][i]}\n### Answer: {examples['response'][i]}"
        texts.append(text)

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train alpha coefficients for MoE models")
    parser.add_argument("--moe_model_path", type=str, default='/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250',
                       help="Path to the MoE model checkpoint")
    parser.add_argument("--safe_data_path", type=str, 
                       default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/gpt4o_moe_model_all_XL_22apater_top2_advbench_data.json",
                       help="Path to the safe evaluation data")
    parser.add_argument("--output_dir", type=str, default="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/alpha_moe_trained",
                       help="Output directory for trained model")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for alpha parameters")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--alpaca_percentage", type=float, default=0.2,
                       help="Percentage of Alpaca dataset to use")
    parser.add_argument("--use_deepspeed", action="store_true",
                       help="Use DeepSpeed for memory optimization")
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization to save memory")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=256,
                       help="Maximum sequence length (reduce to save memory)")
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                    help="Local rank for distributed training (used by DeepSpeed)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.moe_model_path):
        raise FileNotFoundError(f"MoE model path does not exist: {args.moe_model_path}")

    if not os.path.exists(args.safe_data_path):
        raise FileNotFoundError(f"Safe data path does not exist: {args.safe_data_path}")

    # Load the MoE model with memory optimizations
    logger.info(f"Loading MoE model from {args.moe_model_path}")

    # Configure quantization if requested
    quantization_config = None
    if args.use_quantization:
        logger.info("Using 4-bit quantization for memory optimization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    try:
        # Check if using DeepSpeed
        using_deepspeed = hasattr(args, 'deepspeed') and args.deepspeed is not None

        if using_deepspeed:
            # For DeepSpeed, don't move to device manually - let DeepSpeed handle it
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
            }
            device = None  # DeepSpeed will handle device placement
        else:
            # Use single device for non-DeepSpeed training
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
            }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        model = MistralForCausalLM.from_pretrained(
            args.moe_model_path,
            **model_kwargs
        )

        # Move model to single device only if not using DeepSpeed
        if not using_deepspeed and device is not None:
            model = model.to(device)
        logger.info("MoE model loaded successfully")

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to load MoE model: {e}")
        raise
    
    # Wrap model with alpha coefficients
    logger.info("Creating alpha-enhanced MoE model...")
    alpha_model = AlphaMoEModel(model)

    # Move alpha model to the same device as the base model (only if not using DeepSpeed)
    if not using_deepspeed and device is not None:
        alpha_model = alpha_model.to(device)

    # Check if any MoE layers were found
    if not alpha_model.alpha_layers:
        logger.warning("No MoE layers found in the model. Please check if the model is a proper MoE model.")
        logger.info("Available layer names:")
        for name, _ in model.named_modules():
            if any(target in name for target in ["gate_proj", "up_proj", "down_proj"]):
                logger.info(f"  - {name}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create mixed dataset
    logger.info("Creating mixed dataset...")
    try:
        dataset = create_mixed_dataset(args.safe_data_path, args.alpaca_percentage)
        if len(dataset['train']) == 0:
            raise ValueError("No training samples found in the dataset")

        # Tokenize the dataset
        logger.info(f"Tokenizing dataset with max_length={args.max_seq_length}...")
        tokenized_dataset = {}
        for split in ['train', 'val']:
            tokenized_dataset[split] = dataset[split].map(
                lambda examples: preprocess_function(examples, tokenizer, max_length=args.max_seq_length),
                batched=True,
                remove_columns=dataset[split].column_names,
                desc=f"Tokenizing {split} dataset"
            )

        # Clear memory after tokenization
        del dataset
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise
    
    # Training arguments with memory optimizations
    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size if not using_deepspeed else max(1, args.batch_size // 2),
        "per_device_eval_batch_size": args.batch_size if not using_deepspeed else max(1, args.batch_size // 2),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 500,
        "save_total_limit": 2,
        "evaluation_strategy": "steps",
        "logging_strategy": "steps",
        "bf16": True,
        "dataloader_drop_last": True,
        "report_to": "wandb",
        "run_name": f"alpha_moe_training_{args.learning_rate}",
        "remove_unused_columns": False,
        "gradient_checkpointing": True if using_deepspeed else False,  # Enable for DeepSpeed
        "dataloader_num_workers": 0,  # Reduce memory usage
        # "max_grad_norm": 1.0 if using_deepspeed else None,
        # "warmup_steps": 100 if using_deepspeed else None,
        # "lr_scheduler_type": "cosine" if using_deepspeed else None,
    }

    # Add DeepSpeed configuration if requested
    if args.use_deepspeed:
        logger.info("Using DeepSpeed for memory optimization")
        training_args_dict["deepspeed"] = "projects/deepspeed_alpha_config.json"
        training_args_dict["fp16"] = False  # DeepSpeed config handles this

    training_args = TrainingArguments(**training_args_dict)
    
    # Create data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
    )

    # Create trainer
    trainer = Trainer(
        model=alpha_model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in alpha_model.parameters())
    trainable_params = sum(p.numel() for p in alpha_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")

    if trainable_params == 0:
        logger.error("No trainable parameters found! Check if alpha coefficients were properly initialized.")
        return

    # Print alpha values before training
    alpha_model.print_alpha_values("Alpha values before training")

    # Train the model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Print alpha values after training
    alpha_model.print_alpha_values("Alpha values after training")

    # Save the trained alpha coefficients
    logger.info("Saving alpha coefficients...")
    alpha_values = {}
    for layer_name, alpha_layer in alpha_model.alpha_layers.items():
        alpha_values[layer_name] = {}
        for adapter_name, alpha in alpha_layer.alpha_coefficients.items():
            alpha_values[layer_name][adapter_name] = alpha.item()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save alpha coefficients
    alpha_file = f"{args.output_dir}/alpha_coefficients.json"
    with open(alpha_file, 'w') as f:
        json.dump(alpha_values, f, indent=2)

    # Save training configuration
    config_file = f"{args.output_dir}/training_config.json"
    training_config = {
        "moe_model_path": args.moe_model_path,
        "safe_data_path": args.safe_data_path,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "alpaca_percentage": args.alpaca_percentage,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "num_alpha_layers": len(alpha_model.alpha_layers)
    }

    with open(config_file, 'w') as f:
        json.dump(training_config, f, indent=2)

    logger.info(f"Training completed successfully!")
    logger.info(f"Alpha coefficients saved to: {alpha_file}")
    logger.info(f"Training configuration saved to: {config_file}")

    # Save the model
    logger.info("Saving trained model...")
    alpha_model.base_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
