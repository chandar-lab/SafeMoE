"""
Improved LoRA MoE Composer that properly saves ALL model weights including base model weights.

Key improvements:
1. Extracts full state_dict BEFORE save_pretrained() to avoid PEFT's adapter-only saving
2. Manually saves the full state_dict to ensure all weights are included
3. Properly handles normalization weights from the base model
"""

import json
import os
import math
import torch
from tqdm import tqdm
from peft import PeftConfig
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import save_file


class ComposeLoraMoeExpertsV2:
    """
    Improved version that properly saves all model weights.
    
    The key difference from the original:
    - Original: Uses model.save_pretrained() which only saves adapter weights when PEFT is detected
    - V2: Manually extracts and saves full state_dict to include all base model weights
    """
    
    def __init__(
        self,
        config,
        model_cls=AutoModelForCausalLM,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    ):
        self.config = config
        self.model_cls = model_cls
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.state_dict = {}
        self.model_config = {}
        self._tied_weights_keys = []

    def _load_base_model(self, model_id):
        """Load the base model without any adapters."""
        try:
            mps = torch.backends.mps.is_available()
        except:
            mps = False
        
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        if config.model_type == "bert":
            model = self.model_cls.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, trust_remote_code=True
            )
        else:
            model = self.model_cls.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map=None if mps else self.device_map,
                trust_remote_code=True,
            )
        return model

    def to_python_object(self, obj):
        """Convert Hydra configs to Python objects for JSON serialization."""
        from omegaconf import DictConfig, ListConfig
        
        if isinstance(obj, DictConfig):
            return {k: self.to_python_object(v) for k, v in obj.items()}
        elif isinstance(obj, ListConfig):
            return [self.to_python_object(item) for item in obj]
        return obj

    def compose(self):
        """
        Compose all the experts into a single unified checkpoint.
        
        IMPROVED: Extracts full state_dict including base model weights.
        """
        n = len(self.config["experts"])
        model = self._load_base_model(self.config["base_model"])
        self.model_config = model.config.to_dict()
        count_total_router_layers = 0

        # Load all adapters
        for ix, expert in enumerate(self.config["experts"]):
            adapter_id = expert["model_id"]
            adapter_config = PeftConfig.from_pretrained(adapter_id)
            adapter_config_ = adapter_config.to_dict()
            
            # Convert config to JSON-serializable format
            for k, v in adapter_config_.items():
                try:
                    json.dumps(v)
                except:
                    adapter_config_[k] = str(v)
            self.config["adapter_configs"].append(adapter_config_)
            
            # Validate router layers match target modules
            if "router_layers" in self.config:
                print(f"router_layers: {self.config.get('router_layers', 'NOT_SET')}")
                print(f"target_modules: {list(adapter_config.target_modules)}")
                assert sorted(self.config["router_layers"]) == sorted(list(adapter_config.target_modules))
            else:
                self.config["router_layers"] = list(adapter_config.target_modules)
            
            # Load the adapter
            model.load_adapter(adapter_id, adapter_name=str(ix))

        if hasattr(model, "_tied_weights_keys"):
            self._tied_weights_keys.extend(model._tied_weights_keys)

        # CRITICAL: Extract FULL state_dict BEFORE any save operations
        # This ensures we get ALL weights including base model weights
        print("\nExtracting full state_dict (including base model weights)...")
        self.state_dict = model.state_dict()
        
        # Verify we have normalization weights
        norm_weights = [k for k in self.state_dict.keys() if 'norm' in k.lower() and 'weight' in k]
        print(f"✓ Found {len(norm_weights)} normalization weights in state_dict")
        if len(norm_weights) > 0:
            print(f"  Examples: {norm_weights[:3]}")

        # Process layers for MoE conversion
        count_router_layers = 0
        count_averaged_layers = 0
        
        for layer_name, param in tqdm(self.state_dict.items()):
            if (
                any([router_layer in layer_name for router_layer in self.config["router_layers"]])
                and "lora" in layer_name
            ):
                # This is a LoRA layer that needs MoE routing
                count_router_layers += 1
                count_total_router_layers += 1
            elif "lora" not in layer_name:
                # This is a base model layer - keep as is
                count_averaged_layers += 1

        print(f"count_averaged_layers : {count_averaged_layers}")
        print(f"count_router_layers : {count_router_layers}")
        print(f"count_total_router_layers : {count_total_router_layers}")

        return self.state_dict

    def save_checkpoint(self, checkpoint_path):
        """
        Save the composed checkpoint with ALL weights.
        
        IMPROVED: Manually saves state_dict to avoid PEFT's adapter-only saving.
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save config
        config = self.config.copy()
        config["router_layers"] = self.to_python_object(self.config["router_layers"])
        config["adapter_configs"] = self.to_python_object(self.config["adapter_configs"])
        json.dump(config, open(f"{checkpoint_path}/config.json", "w"), indent=1)
        
        # Add gate weights to state_dict
        print("\nAdding gate weights to checkpoint...")
        num_experts = len(self.config["experts"])
        gate_count = 0
        
        for layer_name in list(self.state_dict.keys()):
            if any([router_layer in layer_name for router_layer in self.config["router_layers"]]):
                if "lora_A.0.weight" in layer_name:
                    # This is a LoRA layer that needs a gate
                    base_name = layer_name.replace("lora_A.0.weight", "")
                    gate_weight_name = f"{base_name}gate.weight"
                    
                    if gate_weight_name not in self.state_dict:
                        # Get input dimension from lora_A weight
                        lora_a_weight = self.state_dict[layer_name]
                        in_features = lora_a_weight.shape[1]
                        
                        # Create gate weight with small initialization for bf16 stability
                        gate_weight = torch.zeros(num_experts, in_features, dtype=self.torch_dtype)
                        torch.nn.init.normal_(gate_weight, mean=0.0, std=0.01)
                        
                        self.state_dict[gate_weight_name] = gate_weight
                        gate_count += 1
        
        print(f"✓ Added {gate_count} gate weight tensors to checkpoint")
        
        # CRITICAL: Save state_dict directly using safetensors
        # This bypasses PEFT's adapter-only saving mechanism
        print("\nSaving full state_dict (including all base model weights)...")
        save_file(self.state_dict, f"{checkpoint_path}/model.safetensors")
        
        # Verify saved weights
        from safetensors import safe_open
        with safe_open(f"{checkpoint_path}/model.safetensors", framework="pt") as f:
            saved_keys = list(f.keys())
            norm_weights = [k for k in saved_keys if 'norm' in k.lower() and 'weight' in k]
            print(f"✓ Verified: Saved checkpoint contains {len(saved_keys)} total weights")
            print(f"✓ Verified: Saved checkpoint contains {len(norm_weights)} normalization weights")
        
        # Save model config
        with open(f"{checkpoint_path}/config.json", "w") as f:
            json.dump(self.model_config, f, indent=2)
        
        print(f"\ncheckpoint saved at {checkpoint_path}")
        print("✅ All weights saved successfully (including base model weights)")

