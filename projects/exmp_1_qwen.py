'''
Correct one for traning the routers
'''


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import math
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from mergoo.compose_experts import ComposeExperts
import datasets
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, Trainer, TrainerCallback, AutoModelForCausalLM
# from mergoo.models.modeling_mistral import MistralForCausalLM
from mergoo.models.modeling_qwen3 import Qwen3ForCausalLM
from accelerate import Accelerator
import os
from transformers import AutoTokenizer


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['response'])):
        question_key = "prompt" if "prompt" in example else "instruction"
        # return [f"### Question: {q}\n### Answer: {a}" for q, a in zip(example[question_key], example["response"])]
        text = f"### Question: {example[question_key][i]}\n ### Answer: {example['response'][i]}"
        output_texts.append(text)
    return output_texts

def format_to_prompt_completion(example):
        # Determine which key to use for the question
        question_key = "prompt" if "prompt" in example else "instruction"
        return {
            "prompt": f"### Question: {example[question_key]}\n### Answer:",
            "completion": f" {example['response']}"
        }

    



@hydra.main(version_base=None, config_path="conf", config_name="config_qwen")
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
    # Convert all Hydra configs to regular Python objects for JSON serialization
    model_config = {
        "model_type": cfg.model.model_type,
        "num_experts_per_tok": cfg.model.num_experts_per_tok,
        "base_model": cfg.model.base_model,
        "experts": OmegaConf.to_container(cfg.model.experts, resolve=True),
        "router_layers": OmegaConf.to_container(
            cfg.model.get("router_layers", ["gate_proj", "up_proj", "down_proj"]),
            resolve=True
        ),
    }

    # Create checkpoint
    print("Creating model checkpoint...")

    # Check if checkpoint already exists
    checkpoint_exists = os.path.exists(cfg.paths.model_checkpoint) and \
                       os.path.exists(os.path.join(cfg.paths.model_checkpoint, "config.json"))


    expertmerger = ComposeExperts(
        model_config,
        torch_dtype=torch.bfloat16,  # Use explicit dtype instead of "auto"
        model_cls=Qwen3ForCausalLM   # CRITICAL: Use custom Qwen3 model class
    )
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

    model = Qwen3ForCausalLM.from_pretrained(
            cfg.paths.model_checkpoint,
            torch_dtype=torch.bfloat16,  # Match the dtype used in compose
            attn_implementation="eager",
            trust_remote_code=True  # May be needed for custom model
        )
    # model.config.use_cache = False
    # model.generation_config = model.generation_config or type("GenCfg", (), {})()
    # model.generation_config.use_cache = False

    # ============ REINITIALIZE WEIGHTS FOR BF16 STABILITY ============
    print("\n" + "="*60)
    print("REINITIALIZING WEIGHTS FOR BF16 STABILITY...")
    print("="*60)

    gate_weights_reinitialized = []
    norm_weights_reinitialized = []

    # CRITICAL: Reinitialize ALL gate and norm weights for bf16 stability
    for name, param in model.named_parameters():
        with torch.no_grad():
            if 'gate.weight' in name and 'mlp' in name:
                # Router gate weights - small normal initialization
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
                gate_weights_reinitialized.append(name)
            elif 'norm' in name.lower() and 'weight' in name:
                # ALL normalization weights - initialize to 1.0
                # This fixes the issue where PyTorch randomly initializes missing weights
                param.fill_(1.0)
                norm_weights_reinitialized.append(name)

    print(f"✓ Reinitialized {len(gate_weights_reinitialized)} gate weight tensors")
    print(f"✓ Reinitialized {len(norm_weights_reinitialized)} normalization weight tensors")

    # ============ CHECK FOR CORRUPTED WEIGHTS ============
    print("\n" + "="*60)
    print("CHECKING FOR CORRUPTED WEIGHTS...")
    print("="*60)

    nan_weights = []
    inf_weights = []
    fixed_weights = []

    for name, param in model.named_parameters():
        has_nan = torch.isnan(param).any()
        has_inf = torch.isinf(param).any()

        if has_nan:
            nan_weights.append(name)
        if has_inf:
            inf_weights.append(name)

        if has_nan or has_inf:
            # Reinitialize corrupted weights
            with torch.no_grad():
                if 'norm' in name.lower():
                    # RMSNorm weights should be initialized to 1.0
                    param.fill_(1.0)
                    fixed_weights.append((name, 'RMSNorm -> 1.0'))
                elif 'gate.weight' in name:
                    # Router gate weights - small normal distribution (already done above, but just in case)
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                    fixed_weights.append((name, 'Gate -> N(0, 0.01)'))
                elif 'lora_A' in name:
                    # LoRA A weights - kaiming uniform
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    fixed_weights.append((name, 'LoRA_A -> Kaiming'))
                elif 'lora_B' in name:
                    # LoRA B weights - zeros
                    param.zero_()
                    fixed_weights.append((name, 'LoRA_B -> Zeros'))
                else:
                    # Other weights - xavier uniform
                    if param.dim() >= 2:
                        torch.nn.init.xavier_uniform_(param)
                        fixed_weights.append((name, 'Other -> Xavier'))
                    else:
                        param.zero_()
                        fixed_weights.append((name, 'Other -> Zeros'))

    if nan_weights:
        print(f"\n🚨 Found {len(nan_weights)} weights with NaN")
    if inf_weights:
        print(f"🚨 Found {len(inf_weights)} weights with Inf")

    if fixed_weights:
        print(f"\n✓ Fixed {len(fixed_weights)} corrupted weights:")
        for name, method in fixed_weights[:10]:
            print(f"   - {name}: {method}")
        if len(fixed_weights) > 10:
            print(f"   ... and {len(fixed_weights) - 10} more")
    else:
        print("\n✓ No corrupted weights found")

    # Verify fix
    print("\nVerifying fix...")
    nan_count = sum(1 for _, p in model.named_parameters() if torch.isnan(p).any())
    inf_count = sum(1 for _, p in model.named_parameters() if torch.isinf(p).any())

    if nan_count == 0 and inf_count == 0:
        print("✓ All weights are now valid!")
    else:
        print(f"⚠️  Still have {nan_count} NaN and {inf_count} Inf weights")
        print("\nWeights still with NaN:")
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f"   - {name}: {torch.isnan(p).sum().item()} NaN values out of {p.numel()}")
                if torch.isnan(p).sum().item() <= 10:
                    print(f"     Values: {p.flatten()[:20]}")

    print("="*60 + "\n")
    # ============================================================

    # ============ TEST IF ROUTER IS CAUSING NaN ============
    print("\n" + "="*60)
    print("TESTING IF ROUTER GATES CAUSE NaN...")
    print("="*60)

    # Check router gate statistics
    print("\nRouter gate statistics:")
    gate_count = 0
    for name, param in model.named_parameters():
        if 'gate.weight' in name:
            gate_count += 1
            if gate_count <= 3:
                print(f"\n{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {param.mean().item():.6f}")
                print(f"  Std: {param.std().item():.6f}")
                print(f"  Min: {param.min().item():.6f}")
                print(f"  Max: {param.max().item():.6f}")
                print(f"  Has NaN: {torch.isnan(param).any().item()}")
                print(f"  Has Inf: {torch.isinf(param).any().item()}")

    print(f"\nTotal router gates: {gate_count}")
    print("="*60 + "\n")
    # ============================================================

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


    ############ debugging:
    # After: model = composer.model (or wherever you load the model)

    # ============================================================
    # FREEZE EVERYTHING EXCEPT ROUTER GATES
    # ============================================================
    print("\n" + "="*60)
    print("CONFIGURING TRAINABLE PARAMETERS...")
    print("="*60)

    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    print("✓ Froze all parameters")

    # Step 2: Unfreeze only router gates
    router_params = []
    for name, param in model.named_parameters():
        if 'gate.weight' in name:
            param.requires_grad = True
            router_params.append((name, param.numel()))

    print(f"✓ Unfroze {len(router_params)} router gates")

    # Step 3: Verify
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"\nTrainable: {trainable_params:,} ({trainable_pct:.4f}%)")
    print(f"Total: {total_params:,}")

    # Show first 10 trainable parameters
    print("\nTrainable parameters (first 10):")
    for name, count in router_params[:10]:
        print(f"  ✓ {name}: {count:,} params")

    # Sanity check
    if trainable_pct > 1.0:
        print(f"\n⚠️  WARNING: {trainable_pct:.2f}% trainable (expected < 1%)")
    else:
        print(f"\n✓ Configuration correct: Only {trainable_pct:.4f}% trainable")

    print("="*60 + "\n")
    # ============================================================
    # ============================================================
    # INITIALIZE ROUTER GATES PROPERLY
    # ============================================================
    print("\n" + "="*60)
    print("INITIALIZING ROUTER GATE WEIGHTS...")
    print("="*60)

    import torch.nn as nn

    gate_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'gate') and isinstance(module.gate, nn.Linear):
            # Check current gate statistics
            gate_weight = module.gate.weight
            print(f"\nBefore init - {name}.gate:")
            print(f"  Shape: {gate_weight.shape}")
            print(f"  Mean: {gate_weight.mean().item():.6f}, Std: {gate_weight.std().item():.6f}")
            print(f"  Range: [{gate_weight.min().item():.6f}, {gate_weight.max().item():.6f}]")
            print(f"  Has NaN: {torch.isnan(gate_weight).any().item()}")
            print(f"  Has Inf: {torch.isinf(gate_weight).any().item()}")
            
            # Initialize with small random values (important for routing stability)
            # Option 1: Small normal initialization (RECOMMENDED)
            nn.init.normal_(module.gate.weight, mean=0.0, std=0.01)
            
            # Option 2: Xavier uniform (alternative)
            # nn.init.xavier_uniform_(module.gate.weight, gain=0.01)
            
            # Option 3: Zeros (start with uniform routing)
            # nn.init.zeros_(module.gate.weight)
            
            gate_weight = module.gate.weight
            print(f"After init - {name}.gate:")
            print(f"  Mean: {gate_weight.mean().item():.6f}, Std: {gate_weight.std().item():.6f}")
            print(f"  Range: [{gate_weight.min().item():.6f}, {gate_weight.max().item():.6f}]")
            
            gate_count += 1
            if gate_count >= 3:  # Only show first 3 for brevity
                print(f"\n... and {len(router_params) - 3} more gates")
                break

    print(f"\n✓ Initialized {len(router_params)} router gates")
    print("="*60 + "\n")
    # ============================================================
    ######################



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

    train_dataset = dataset['train'].map(format_to_prompt_completion, remove_columns=dataset['train'].column_names)
    eval_dataset = dataset['val'].map(format_to_prompt_completion, remove_columns=dataset['val'].column_names)
    test_dataset = dataset['test'].map(format_to_prompt_completion, remove_columns=dataset['test'].column_names)

    print(f"Dataset sizes - Train: {len(dataset['train'])}, Val: {len(dataset['val'])}, Test: {len(dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model, trust_remote_code=True)
    # If the tokenizer has no pad token, set one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ############ Debugging:
    print("\n" + "="*60)
    print("COMPREHENSIVE NaN DIAGNOSTIC")
    print("="*60)

    # Step 1: Check all model weights for NaN/Inf
    print("\n1. CHECKING MODEL WEIGHTS FOR NaN/Inf...")
    nan_weights = []
    inf_weights = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_weights.append(name)
        if torch.isinf(param).any():
            inf_weights.append(name)

    if nan_weights:
        print(f"🚨 FOUND {len(nan_weights)} WEIGHTS WITH NaN:")
        for name in nan_weights[:10]:
            print(f"   - {name}")
    else:
        print("✓ No NaN in model weights")

    if inf_weights:
        print(f"🚨 FOUND {len(inf_weights)} WEIGHTS WITH Inf:")
        for name in inf_weights[:10]:
            print(f"   - {name}")
    else:
        print("✓ No Inf in model weights")

    # Step 2: Check LoRA weights specifically
    print("\n2. CHECKING LORA WEIGHTS...")
    lora_stats = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_stats.append({
                'name': name,
                'shape': param.shape,
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'has_nan': torch.isnan(param).any().item(),
                'has_inf': torch.isinf(param).any().item(),
            })

    if lora_stats:
        print(f"Found {len(lora_stats)} LoRA parameters")
        print("\nFirst 3 LoRA weight statistics:")
        for i, stats in enumerate(lora_stats[:3]):
            print(f"\n{i+1}. {stats['name']}")
            print(f"   Shape: {stats['shape']}")
            print(f"   Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"   Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            if stats['has_nan']:
                print(f"   🚨 HAS NaN VALUES!")
            if stats['has_inf']:
                print(f"   🚨 HAS Inf VALUES!")

    # Step 3: Test with simple input
    print("\n3. TESTING FORWARD PASS WITH SIMPLE INPUT...")
    model.eval()

    # Create simple test input
    simple_input = torch.randint(0, 1000, (1, 5)).to(model.device)
    print(f"Simple input shape: {simple_input.shape}")
    print(f"Simple input: {simple_input}")

    # Add hooks to detect where NaN is introduced
    nan_detected_in = []

    def check_nan_hook(module, input, output):
        module_name = module.__class__.__name__
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                nan_detected_in.append(f"{module_name}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                    nan_detected_in.append(f"{module_name}[{i}]")

    # Register hooks on key modules
    hooks = []
    for name, module in model.named_modules():
        if any(x in name for x in ['norm', 'gate', 'mlp', 'attn']):
            hooks.append(module.register_forward_hook(check_nan_hook))

    with torch.no_grad():
        try:
            simple_out = model(simple_input)
            print(f"✓ Simple forward pass succeeded")
            print(f"  Output logits shape: {simple_out.logits.shape}")
            print(f"  Logits has NaN: {torch.isnan(simple_out.logits).any().item()}")
            print(f"  Logits has Inf: {torch.isinf(simple_out.logits).any().item()}")
            print(f"  Logits range: [{simple_out.logits.min().item():.4f}, {simple_out.logits.max().item():.4f}]")

            if nan_detected_in:
                print(f"\n🚨 NaN first detected in these modules:")
                for module in nan_detected_in[:10]:
                    print(f"   - {module}")
                if len(nan_detected_in) > 10:
                    print(f"   ... and {len(nan_detected_in) - 10} more")
        except Exception as e:
            print(f"🚨 Simple forward pass FAILED: {e}")
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

    # Step 4: Test with labels
    print("\n4. TESTING FORWARD PASS WITH LABELS...")
    simple_labels = simple_input.clone()

    with torch.no_grad():
        try:
            simple_out_with_loss = model(simple_input, labels=simple_labels)
            loss = simple_out_with_loss.loss
            print(f"✓ Forward pass with labels succeeded")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Loss is NaN: {torch.isnan(loss).item()}")
            print(f"  Loss is Inf: {torch.isinf(loss).item()}")
        except Exception as e:
            print(f"🚨 Forward pass with labels FAILED: {e}")

    # Step 5: Test with actual data
    print("\n5. TESTING WITH ACTUAL DATASET...")
    sample = train_dataset[0]
    print(f"SAMPLE KEYS: {sample.keys()}")
    print(f"PROMPT: {sample['prompt'][:100]}...")
    print(f"COMPLETION: {sample['completion'][:100]}...")

    # Build a small batch
    batch_texts = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    texts = [b['prompt'] + b['completion'] for b in batch_texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Manually build labels for completion_only_loss
    prompt_lens = [len(tokenizer(b['prompt'])['input_ids']) for b in batch_texts]
    labels = inputs['input_ids'].clone()
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100

    print(f"\nInput shape: {inputs['input_ids'].shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of non-masked tokens: {(labels != -100).sum().item()}")
    print(f"Are all labels -100?: {(labels == -100).all().item()}")

    inputs['labels'] = labels.to(model.device)
    inputs = {k: v.to(model.device) for k,v in inputs.items()}

    with torch.no_grad():
        try:
            out = model(**inputs)
            loss = out.loss
            print(f"\n✓ Forward pass with real data succeeded")
            print(f"  Loss: {loss.item() if not torch.isnan(loss) else 'NaN'}")
            print(f"  Loss is NaN: {torch.isnan(loss).item()}")
            print(f"  Loss is Inf: {torch.isinf(loss).item()}")
            print(f"  Logits has NaN: {torch.isnan(out.logits).any().item()}")
            print(f"  Logits has Inf: {torch.isinf(out.logits).any().item()}")

            if torch.isnan(loss):
                print("\n🚨 LOSS IS NaN! Checking intermediate values...")
                # Check if it's a label masking issue
                if (labels != -100).sum() == 0:
                    print("   🚨 ALL LABELS ARE MASKED (-100)! No loss can be computed.")
                else:
                    print(f"   ✓ {(labels != -100).sum().item()} tokens are not masked")

        except Exception as e:
            print(f"🚨 Forward pass with real data FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("="*60 + "\n")
    #####################

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
    trainer_args = SFTConfig(
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
        fp16=True,
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
        max_length=512,
        packing=False,
        completion_only_loss=True,
        max_grad_norm=1.0,

        # CRITICAL: Fix learning rate scheduler
        # By default, Trainer uses linear warmup + decay which reduces LR to ~1e-7
        # For router training, we want constant LR
        lr_scheduler_type="constant",  # No warmup, no decay - constant LR
        warmup_steps=0,  # No warmup period

    )

    # Create trainer
    trainer = SFTTrainer(
        model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # formatting_func=formatting_prompts_func,
        # max_seq_length=cfg.training.max_seq_length,
        processing_class=tokenizer,
        callbacks=[WandbCallback()] if cfg.wandb.enabled else []
    )


  # === TRAINING ARGS ===

    # training_args = SFTConfig(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=2,
    #     num_train_epochs=4,
    #     learning_rate=2e-4,
    #     logging_steps=10,
    #     save_steps=100,
    #     # evaluation_strategy="no",
    #     bf16=torch.cuda.is_bf16_supported(),
    #     fp16=not torch.cuda.is_bf16_supported(),
    #     report_to="wandb",  # Enable wandb logging
    #     # run_name=run_name,
    #     logging_first_step=True,
    #     max_length=512,
    #     packing=False,
    #     # APPROACH 1 (CURRENT): With prompt-completion format
    #     # completion_only_loss=True automatically computes loss only on the completion part
    #     completion_only_loss=True,

    #     # APPROACH 2 (ALTERNATIVE - COMMENTED OUT): With language modeling format
    #     # If using the text field approach (see APPROACH 2 above), use these settings:
    #     # dataset_text_field="text",
    #     # completion_only_loss=False,  # Compute loss on entire sequence
    # )

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

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     # tokenizer=tokenizer,
    #     processing_class=tokenizer,
    #     args=training_args,
    # )





    # Ensure evaluation is properly configured
    # print(f"Evaluation strategy: {trainer_args.evaluation_strategy}")
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

