"""
Improved Qwen3 MoE training script using ComposeLoraMoeExpertsV2.

Key improvements:
1. Uses V2 composer that properly saves all model weights
2. No need to reinitialize normalization weights after loading
3. Cleaner and more reliable workflow
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Import the improved V2 composer
from mergoo.composers.composer_lora_moe_v2 import ComposeLoraMoeExpertsV2


@hydra.main(version_base=None, config_path="../configs", config_name="qwen3_moe")
def main(cfg: DictConfig):
    print("="*80)
    print("QWEN3 MOE WITH LORA - IMPROVED VERSION (V2)")
    print("="*80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # ============ STEP 1: COMPOSE MOE MODEL ============
    print("\n" + "="*60)
    print("STEP 1: COMPOSING MOE MODEL WITH LORA EXPERTS")
    print("="*60)
    
    # Prepare config for composer
    composer_config = {
        "base_model": cfg.model.base_model,
        "experts": [
            {"model_id": expert_id} 
            for expert_id in cfg.model.expert_models
        ],
        "router_layers": OmegaConf.to_container(
            cfg.model.get("router_layers", ["gate_proj", "up_proj", "down_proj"]), 
            resolve=True
        ),
        "adapter_configs": [],
    }
    
    checkpoint_path = cfg.model.checkpoint_path
    
    # Check if checkpoint already exists
    import os
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Checkpoint already exists at: {checkpoint_path}")
        print("Skipping composition step...")
    else:
        print(f"\nComposing MoE model...")
        print(f"Base model: {cfg.model.base_model}")
        print(f"Experts: {cfg.model.expert_models}")
        print(f"Router layers: {composer_config['router_layers']}")
        
        # Use the improved V2 composer
        composer = ComposeLoraMoeExpertsV2(
            config=composer_config,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        
        # Compose and save
        composer.compose()
        composer.save_checkpoint(checkpoint_path)
        
        print(f"\n✅ MoE model composed and saved to: {checkpoint_path}")
    
    # ============ STEP 2: LOAD MODEL FOR TRAINING ============
    print("\n" + "="*60)
    print("STEP 2: LOADING MOE MODEL FOR TRAINING")
    print("="*60)
    
    from transformers import AutoModelForCausalLM
    
    print(f"Loading model from: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
    
    # ============ VERIFY WEIGHTS ============
    print("\n" + "="*60)
    print("VERIFYING MODEL WEIGHTS")
    print("="*60)
    
    # Check for NaN/Inf in weights
    nan_count = 0
    inf_count = 0
    total_params = 0
    norm_weights = []
    gate_weights = []
    
    for name, param in model.named_parameters():
        total_params += 1
        if torch.isnan(param).any():
            nan_count += 1
            print(f"  ⚠️  NaN found in: {name}")
        if torch.isinf(param).any():
            inf_count += 1
            print(f"  ⚠️  Inf found in: {name}")
        
        if 'norm' in name.lower() and 'weight' in name:
            norm_weights.append(name)
        if 'gate.weight' in name and 'mlp' in name:
            gate_weights.append(name)
    
    if nan_count == 0 and inf_count == 0:
        print(f"✅ All {total_params} parameters are clean (no NaN/Inf)")
    else:
        print(f"❌ Found {nan_count} NaN and {inf_count} Inf parameters out of {total_params}")
        print("\n⚠️  WARNING: Model has corrupted weights!")
        print("This should NOT happen with V2 composer.")
        print("If you see this, there may be an issue with the checkpoint.")
    
    print(f"\n✓ Found {len(norm_weights)} normalization weights")
    print(f"✓ Found {len(gate_weights)} gate weights")
    
    # Sample a few normalization weights to verify they're loaded correctly
    if len(norm_weights) > 0:
        print("\nSample normalization weight values:")
        for name in norm_weights[:3]:
            param = dict(model.named_parameters())[name]
            print(f"  {name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    
    # ============ STEP 3: PREPARE TRAINING ============
    print("\n" + "="*60)
    print("STEP 3: PREPARING TRAINING")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"\nLoading dataset: {cfg.data.dataset_name}")
    dataset = load_dataset(cfg.data.dataset_name, split=cfg.data.split)
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        bf16=True,  # Use bfloat16
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        warmup_steps=cfg.training.warmup_steps,
        report_to=cfg.training.get("report_to", "none"),
    )
    
    # ============ STEP 4: TRAIN ============
    print("\n" + "="*60)
    print("STEP 4: STARTING TRAINING")
    print("="*60)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.data.max_seq_length,
        dataset_text_field=cfg.data.text_field,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to: {cfg.training.output_dir}/final")
    trainer.save_model(f"{cfg.training.output_dir}/final")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

