from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import prepare_model_for_kbit_training
import json
import os
from datetime import datetime
import wandb

# === Load base model and tokenizer ===
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
base_model.eval()  # base is always frozen

# === Load frozen LoRA_1 ===
lora1_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-drug_abuse"
model_with_lora1 = PeftModel.from_pretrained(base_model, lora1_path)
model_with_lora1.eval()  # freeze LoRA_1

# === Create trainable LoRA_2 ===
lora_config2 = LoraConfig(
    r=32, lora_alpha=64, target_modules=["up_proj", "down_proj", "gate_proj"], lora_dropout=0.1,
    bias="none", task_type=TaskType.CAUSAL_LM
)
model_with_lora2 = get_peft_model(base_model, lora_config2)
model_with_lora2.train()

# === Discriminator (to distinguish h1 vs. h2) === its output is a scalar
class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # Reduce size to prevent overfitting
            nn.LayerNorm(hidden_size // 4),  # Add layer norm for stability
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_size // 4, 1)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for stability
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, h):
        return self.fc(h)

discriminator = Discriminator(hidden_size=4096).cuda()  # Keep in float32 for stability

# Debug: Print dtypes to verify consistency
print(f"Base model dtype: {next(base_model.parameters()).dtype}")
print(f"LoRA1 model dtype: {next(model_with_lora1.parameters()).dtype}")
print(f"LoRA2 model dtype: {next(model_with_lora2.parameters()).dtype}")
print(f"Discriminator dtype: {next(discriminator.parameters()).dtype}")

# === Initialize Weights & Biases ===
wandb.init(
    project="adversarial-training",
    name=f"lora2-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    config={
        "model_id": model_id,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "batch_size": 4,
        "epochs": 10,
        "lr_lora2": 1e-5,
        "lr_discriminator": 5e-5,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "adversarial_weight": 0.1,
        "max_eval_samples": 5,
        "dataset_path": "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substance.json"
    }
)

# === Dataset & Dataloader ===
# Option 1: Load from local JSON file
dataset_path = "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substance.json"
dataset = load_dataset("json", data_files=dataset_path)["train"]

# Option 2: Load HuggingFace dataset
# dataset = load_dataset("fwnlp/self-instruct-safety-alignment")["train"]
train_test = dataset.train_test_split(test_size=0.1, seed=42)
if len(train_test['train']) < 6000:
    train_dataset = train_test['train'].shuffle(seed=42).select(range(len(train_test['train'])))
else:
    train_dataset = train_test['train'].shuffle(seed=42).select(range(6000))
if len(train_test['test']) < 1000:
    eval_dataset = train_test['test'].shuffle(seed=42).select(range(len(train_test['test'])))
else:
    eval_dataset = train_test['test'].shuffle(seed=42).select(range(1000))

def formatting_func(example):
    # Determine which key to use for single example
    question_key = "prompt" if "prompt" in example else "instruction"
    question = example[question_key]
    answer = example["response"]
    return f"### Question: {question}\n### Answer: {answer}"

def collate_fn(batch):
    # Apply formatting and tokenize in one step
    formatted_texts = [formatting_func(example) for example in batch]
    inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True)
    return inputs

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# === Evaluation function ===
def evaluate_model_generation(model, tokenizer, eval_dataset, epoch, max_samples=10):
    """
    Generate sentences from the model using eval dataset prompts
    """
    model.eval()
    results = []

    # Take a subset of eval data for generation
    eval_samples = eval_dataset.shuffle(seed=42).select(range(min(max_samples, len(eval_dataset))))

    # Create wandb table for this epoch's generations
    columns = ["epoch", "sample_id", "prompt", "ground_truth", "generated_answer"]
    wandb_table = wandb.Table(columns=columns)

    for i, example in enumerate(eval_samples):
        # Get the prompt
        question_key = "prompt" if "prompt" in example else "instruction"
        prompt = example[question_key]
        ground_truth = example["response"]

        # Format the prompt for generation
        formatted_prompt = f"### Question: {prompt}\n### Answer:"

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated answer (remove the prompt part)
        generated_answer = generated_text[len(formatted_prompt):].strip()

        result = {
            "epoch": epoch,
            "sample_id": i,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "full_generated_text": generated_text
        }
        results.append(result)

        # Add to wandb table
        wandb_table.add_data(epoch, i, prompt[:100], ground_truth[:100], generated_answer[:100])

        print(f"Epoch {epoch}, Sample {i+1}/{max_samples}: Generated response for prompt: {prompt[:50]}...")

    # Log the table to wandb
    wandb.log({f"generated_samples_epoch_{epoch}": wandb_table}, step=epoch)

    model.train()  # Set back to training mode
    return results


# === Optimizers ===
# Reduced learning rates for stability
optimizer_lora2 = torch.optim.AdamW(model_with_lora2.parameters(), lr=1e-5, weight_decay=0.01)
optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=5e-5, weight_decay=0.01)

# === Training Loop ===
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

# Initialize results storage
all_evaluation_results = []
output_file = "lora2_evaluation_results.json"

# Training metrics tracking
step = 0
nan_batch_count = 0
max_nan_batches = 10  # Reset discriminator if too many NaN batches

for epoch in range(10):
    epoch_kl_losses = []
    epoch_adv_losses = []
    epoch_disc_losses = []
    for batch in dataloader:
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        with torch.no_grad():
            out_base = base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            out_lora1 = model_with_lora1(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        out_lora2 = model_with_lora2(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # === Task: KL divergence from base logits ===
        logits_base = out_base.logits
        logits_lora2 = out_lora2.logits

        # Debug: Check for NaN/Inf in logits
        if torch.isnan(logits_base).any() or torch.isinf(logits_base).any():
            print("WARNING: NaN/Inf detected in base logits!")
        if torch.isnan(logits_lora2).any() or torch.isinf(logits_lora2).any():
            print("WARNING: NaN/Inf detected in lora2 logits!")

        # Use more stable KL divergence computation
        # Convert to float32 for numerical stability, then back to float16
        logits_base_stable = logits_base.float()
        logits_lora2_stable = logits_lora2.float()

        log_probs_base = F.log_softmax(logits_base_stable, dim=-1)
        log_probs_lora2 = F.log_softmax(logits_lora2_stable, dim=-1)

        # KL(P||Q) = sum(P * (log(P) - log(Q)))
        # Here we want KL(lora2||base)
        kl_loss = F.kl_div(log_probs_base, log_probs_lora2.exp(), reduction='batchmean')

        # Convert back to float16 if needed
        kl_loss = kl_loss.half() if logits_base.dtype == torch.float16 else kl_loss

        # === Adversarial loss ===
        # h1 = out_lora1.hidden_states[-1][:, 0, :]  # [CLS] or first token
        # h2 = out_lora2.hidden_states[-1][:, 0, :]

        # === Extract hidden states for adversarial loss (last non-pad token) ===
        h1_all = out_lora1.hidden_states[-1]
        h2_all = out_lora2.hidden_states[-1]

        last_token_indices = attention_mask.sum(dim=1) - 1  # [batch]
        h1 = h1_all[range(h1_all.size(0)), last_token_indices]  # [batch, hidden]
        h2 = h2_all[range(h2_all.size(0)), last_token_indices]

        # Debug: Check for NaN/Inf in hidden states before discriminator
        if torch.isnan(h1).any() or torch.isinf(h1).any():
            print("WARNING: NaN/Inf detected in h1 hidden states!")
            print(f"h1 stats: min={h1.min()}, max={h1.max()}, mean={h1.mean()}")
        if torch.isnan(h2).any() or torch.isinf(h2).any():
            print("WARNING: NaN/Inf detected in h2 hidden states!")
            print(f"h2 stats: min={h2.min()}, max={h2.max()}, mean={h2.mean()}")

        # Convert to float32 for discriminator processing (more stable than float16)
        h1 = h1.float()
        h2 = h2.float()

        # Only clamp if there are extreme values, but don't normalize
        # This preserves the natural differences between h1 and h2
        h1 = torch.clamp(h1, min=-50.0, max=50.0)
        h2 = torch.clamp(h2, min=-50.0, max=50.0)

        # Skip adversarial training if hidden states are problematic
        if torch.isnan(h1).any() or torch.isinf(h1).any() or torch.isnan(h2).any() or torch.isinf(h2).any():
            print("WARNING: Problematic hidden states detected. Skipping adversarial training for this batch.")
            nan_batch_count += 1

            # Reset discriminator if too many NaN batches
            if nan_batch_count >= max_nan_batches:
                print(f"WARNING: Too many NaN batches ({nan_batch_count}). Reinitializing discriminator.")
                discriminator = Discriminator(hidden_size=4096).cuda()  # Keep in float32
                optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=5e-5, weight_decay=0.01)
                nan_batch_count = 0

            # Only do KL loss training
            optimizer_lora2.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_with_lora2.parameters(), max_norm=1.0)
            optimizer_lora2.step()

            print(f"Epoch {epoch} | KL: {kl_loss.item():.4f} | ADV: skipped | D Loss: skipped")
            continue

        # Train discriminator
        logits_disc_h1 = discriminator(h1.detach())
        logits_disc_h2 = discriminator(h2.detach())

        # Debug: Check for NaN/Inf in discriminator outputs
        if torch.isnan(logits_disc_h1).any() or torch.isinf(logits_disc_h1).any():
            print("WARNING: NaN/Inf detected in discriminator output for h1!")
            print(f"h1 input stats: min={h1.min()}, max={h1.max()}, mean={h1.mean()}")
        if torch.isnan(logits_disc_h2).any() or torch.isinf(logits_disc_h2).any():
            print("WARNING: NaN/Inf detected in discriminator output for h2!")
            print(f"h2 input stats: min={h2.min()}, max={h2.max()}, mean={h2.mean()}")

        # If discriminator outputs are still problematic, skip adversarial training
        if (torch.isnan(logits_disc_h1).any() or torch.isinf(logits_disc_h1).any() or
            torch.isnan(logits_disc_h2).any() or torch.isinf(logits_disc_h2).any()):
            print("WARNING: Discriminator outputs are problematic. Skipping adversarial training for this batch.")
            nan_batch_count += 1

            # Reset discriminator if too many NaN batches
            if nan_batch_count >= max_nan_batches:
                print(f"WARNING: Too many NaN batches ({nan_batch_count}). Reinitializing discriminator.")
                discriminator = Discriminator(hidden_size=4096).cuda()  # Keep in float32
                optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=5e-5, weight_decay=0.01)
                nan_batch_count = 0

            # Only do KL loss training
            optimizer_lora2.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_with_lora2.parameters(), max_norm=1.0)
            optimizer_lora2.step()

            print(f"Epoch {epoch} | KL: {kl_loss.item():.4f} | ADV: skipped | D Loss: skipped")
            continue

        # Discriminator is already in float32, so no conversion needed
        # Create target tensors
        zeros_target = torch.zeros_like(logits_disc_h1)
        ones_target = torch.ones_like(logits_disc_h2)

        # Compute losses
        loss_disc_1 = F.binary_cross_entropy_with_logits(logits_disc_h1, zeros_target)
        loss_disc_2 = F.binary_cross_entropy_with_logits(logits_disc_h2, ones_target)
        loss_disc = loss_disc_1 + loss_disc_2

        # Check for NaN in discriminator loss before backward
        if torch.isnan(loss_disc).any():
            print("WARNING: NaN detected in discriminator loss! Skipping this batch.")
            continue

        optimizer_disc.zero_grad()
        loss_disc.backward()

        # Gradient clipping for discriminator
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

        optimizer_disc.step()

        # Train LoRA₂ to fool discriminator
        logits_disc_h2_adv = discriminator(h2)

        # Discriminator is already in float32, so no conversion needed
        zeros_target_adv = torch.zeros_like(logits_disc_h2_adv)
        adv_loss = F.binary_cross_entropy_with_logits(logits_disc_h2_adv, zeros_target_adv)  # want D to think h2 ≈ h1

        total_loss = kl_loss + 0.1 * adv_loss

        # Check for NaN in total loss before backward
        if torch.isnan(total_loss).any() or torch.isnan(kl_loss).any() or torch.isnan(adv_loss).any():
            print(f"WARNING: NaN detected in losses! KL: {kl_loss.item()}, ADV: {adv_loss.item()}, Total: {total_loss.item()}")
            print("Skipping this batch.")
            continue

        optimizer_lora2.zero_grad()
        total_loss.backward()

        # Gradient clipping for LoRA2
        torch.nn.utils.clip_grad_norm_(model_with_lora2.parameters(), max_norm=1.0)

        optimizer_lora2.step()

        # Store losses for epoch averaging
        epoch_kl_losses.append(kl_loss.item())
        epoch_adv_losses.append(adv_loss.item())
        epoch_disc_losses.append(loss_disc.item())

        # Log to wandb every step
        wandb.log({
            "step_kl_loss": kl_loss.item(),
            "step_adv_loss": adv_loss.item(),
            "step_disc_loss": loss_disc.item(),
            "step_total_loss": total_loss.item(),
            "epoch": epoch,
            "step": step
        }, step=step)

        step += 1

        print(f"Epoch {epoch} | KL: {kl_loss.item():.4f} | ADV: {adv_loss.item():.4f} | D Loss: {loss_disc.item():.4f}")

    # === Log epoch-level metrics to wandb ===
    avg_kl_loss = sum(epoch_kl_losses) / len(epoch_kl_losses)
    avg_adv_loss = sum(epoch_adv_losses) / len(epoch_adv_losses)
    avg_disc_loss = sum(epoch_disc_losses) / len(epoch_disc_losses)

    wandb.log({
        "epoch_avg_kl_loss": avg_kl_loss,
        "epoch_avg_adv_loss": avg_adv_loss,
        "epoch_avg_disc_loss": avg_disc_loss,
        "epoch_avg_total_loss": avg_kl_loss + 0.1 * avg_adv_loss,
        "epoch": epoch
    }, step=epoch)

    # === Evaluation at the end of each epoch ===
    print(f"\n=== Evaluating LoRA2 model at end of epoch {epoch} ===")
    epoch_results = evaluate_model_generation(model_with_lora2, tokenizer, eval_dataset, epoch, max_samples=5)
    all_evaluation_results.extend(epoch_results)

    # Log evaluation metrics to wandb
    wandb.log({
        "num_eval_samples": len(epoch_results),
        "total_eval_samples": len(all_evaluation_results)
    }, step=epoch)

    # Save results to JSON file after each epoch (incremental save)
    with open(output_file, 'w') as f:
        json.dump(all_evaluation_results, f, indent=2)

    print(f"Evaluation results saved to {output_file}")
    print(f"Completed epoch {epoch} | Avg KL: {avg_kl_loss:.4f} | Avg ADV: {avg_adv_loss:.4f} | Avg D: {avg_disc_loss:.4f}")
    print("="*50 + "\n")

print(f"\nTraining completed! All evaluation results saved to {output_file}")
print(f"Total evaluation samples collected: {len(all_evaluation_results)}")

# === Final wandb logging and cleanup ===
wandb.log({
    "training_completed": True,
    "total_epochs": 10,
    "total_eval_samples": len(all_evaluation_results)
})

# Upload the final JSON file as an artifact
artifact = wandb.Artifact("evaluation_results", type="dataset")
artifact.add_file(output_file)
wandb.log_artifact(artifact)

print(f"Results uploaded to wandb. Check your wandb dashboard for detailed metrics and visualizations.")
wandb.finish()
