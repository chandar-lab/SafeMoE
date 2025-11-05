import os
import json
import math
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import PeftConfig
from mergoo.safe_saving import save_pretrained


class ComposeLoraMoeExperts:
    def __init__(
        self,
        config,
        torch_dtype=torch.float16,
        device="cpu",
        device_map="auto",
        max_shard_size="9GB",
        model_cls=AutoModelForCausalLM,
    ):
        """
        Args:
            config (dict): Configuration required to setup the composer. Explore configs/ for examples/
            torch_dtype (torch.dtype, optional): Datatype for loading and saving the weights. Defaults to torch.float16.
            device (str, optional): Defaults to "cpu".
            device_map (str, optional): Defaults to "auto".
            max_shard_size (str, optional): Maximum Shard size checkpoint chuncks. Defaults to "9GB".
            model_cls (type, optional): Change this when using a architecture not registered with transformers. Defaults to AutoModelForCausalLM.
        """
        self.config = config
        self.torch_dtype = torch_dtype
        self._tied_weights_keys = []
        self.device = device
        self.device_map = device_map
        self.max_shard_size = max_shard_size
        self.model_cls = model_cls
        self.config["router_layers_index"] = self.config.get("router_layers_index", [])
        self.moe_layer_index = self.config["router_layers_index"]
        self.select_moe_model_config_idx = 0
        self.config["adapter_configs"] = []
        self._set_moe_layer_index()

    def _set_moe_layer_index(self):
        if len(self.moe_layer_index) == 0:
            self._check_moe_layers = lambda x: True
            print(f"MoE Layer Index : [*]")

        elif len(self.moe_layer_index) >= 1 and self.moe_layer_index[0] is not None:
            self._check_moe_layers = lambda x: x in self.moe_layer_index
            print(f"MoE Layer Index : {self.moe_layer_index}")

        else:
            self._check_moe_layers = lambda x: False
            print(f"No MoE layer indexes.")

    def _is_layer_suitable_for_router(self, layer_identifier, model_layer):
        model_layer_index = [int(x) for x in model_layer.split(".") if x.isdigit()]
        if not model_layer_index:
            valid_layer_index = False
        else:
            if "lora" in model_layer.lower():
                assert len(model_layer_index) == 2
            else:
                assert len(model_layer_index) == 1  # [layer index, adapter index]
            valid_layer_index = self._check_moe_layers(model_layer_index[0])

        if (layer_identifier in model_layer) and valid_layer_index:
            return True
        return False

    def _load_base_model(self, model_id):
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
        if mps:
            return model.to(self.device)
        return model

    def compose(self):
        """
        Compose all the experts into a single unified checkpoint.
        """
        n = len(self.config["experts"])
        model = self._load_base_model(self.config["base_model"])
        self.model_config = model.config.to_dict()
        count_total_router_layers = 0

        for ix, expert in enumerate(self.config["experts"]):
            adapter_id = expert["model_id"]
            adapter_config = PeftConfig.from_pretrained(adapter_id)
            adapter_config_ = adapter_config.to_dict()
            for k, v in adapter_config_.items():
                try:
                    json.dumps(v)
                except:
                    adapter_config_[k] = str(v)
            self.config["adapter_configs"].append(adapter_config_)
            # check if all the lora are having same target modules
            if "router_layers" in self.config:
                print(f"router_layers: {self.config.get('router_layers', 'NOT_SET')}")
                print(f"target_modules: {list(adapter_config.target_modules)}")
                assert sorted(self.config["router_layers"]) == sorted(list(adapter_config.target_modules))
            else:
                self.config["router_layers"] = list(adapter_config.target_modules)
            # load the adapter
            model.load_adapter(adapter_id, adapter_name=str(ix))

        if hasattr(model, "_tied_weights_keys"):
            self._tied_weights_keys.extend(model._tied_weights_keys)

        # Check and fix corrupted weights BEFORE extracting state_dict
        print("\nChecking for corrupted weights in model...")
        nan_count = 0
        inf_count = 0
        fixed_count = 0

        for name, param in model.named_parameters():
            has_nan = torch.isnan(param).any()
            has_inf = torch.isinf(param).any()

            if has_nan or has_inf:
                if has_nan:
                    nan_count += 1
                if has_inf:
                    inf_count += 1

                # Fix corrupted weights IN PLACE
                with torch.no_grad():
                    if 'norm' in name.lower() and 'weight' in name:
                        # RMSNorm/LayerNorm weights should be 1.0
                        param.fill_(1.0)
                        fixed_count += 1
                        print(f"  ✓ Fixed {name}: RMSNorm -> 1.0")
                    elif 'lora_A' in name:
                        # LoRA A weights - reinitialize with kaiming
                        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                        fixed_count += 1
                        print(f"  ✓ Fixed {name}: LoRA_A -> Kaiming")
                    elif 'lora_B' in name:
                        # LoRA B weights - zeros
                        param.zero_()
                        fixed_count += 1
                        print(f"  ✓ Fixed {name}: LoRA_B -> Zeros")
                    else:
                        # Other weights - reinitialize with xavier
                        if param.dim() >= 2:
                            torch.nn.init.xavier_uniform_(param)
                            fixed_count += 1
                            print(f"  ✓ Fixed {name}: Other -> Xavier")
                        else:
                            param.zero_()
                            fixed_count += 1
                            print(f"  ✓ Fixed {name}: Other -> Zeros")

        if nan_count > 0 or inf_count > 0:
            print(f"\n⚠️  Found {nan_count} NaN and {inf_count} Inf weights in model")
            print(f"✓ Fixed {fixed_count} corrupted weights")
        else:
            print("✓ No corrupted weights found in model")

        self.state_dict = model.state_dict()

        count_router_layers = 0
        count_averaged_layers = 0
        for layer_name, param in tqdm(model.state_dict().items()):
            if (
                sum(
                    [
                        self._is_layer_suitable_for_router(router_layer, layer_name)
                        for router_layer in self.config["router_layers"]
                    ]
                )
                == 1
            ):
                # Note: Index of adapter in the config are kept as adapter names, while saving.
                # Similar should be case while loading the adapters
                assert layer_name in self.state_dict
                count_total_router_layers += 1
                count_router_layers += 1
            else:
                assert layer_name in self.state_dict
                count_averaged_layers += 1

        print(f"count_averaged_layers : {count_averaged_layers}")
        print(f"count_router_layers : {count_router_layers}")
        print(f"count_total_router_layers : {count_total_router_layers}")
        del model
        gc.collect()

    def save_checkpoint(self, checkpoint_path):
        """
        Save the composed Unified checkpoint.
        Checkpoints are saved as safe tensors in shards(chuncks).
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        config = self.model_config
        config["num_experts"] = len(self.config["experts"])
        config["num_experts_per_tok"] = self.config["num_experts_per_tok"]

        # Convert Hydra ListConfig/DictConfig to regular Python objects for JSON serialization
        def to_python_object(obj):
            """Recursively convert Hydra configs to Python objects"""
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                if hasattr(obj, 'items'):  # Dict-like
                    return {k: to_python_object(v) for k, v in obj.items()}
                else:  # List-like
                    return [to_python_object(item) for item in obj]
            return obj

        config["router_layers"] = to_python_object(self.config["router_layers"])
        config["adapter_configs"] = to_python_object(self.config["adapter_configs"])

        layer_indexes = list(range(config["num_hidden_layers"]))
        if not self.config["router_layers_index"]:
            config["router_layers_index"] = layer_indexes

        else:
            config["router_layers_index"] = list(
                set(layer_indexes).intersection(set(self.config["router_layers_index"]))
            )

        json.dump(config, open(f"{checkpoint_path}/config.json", "w"), indent=1)

        # Add gate weights to state_dict with proper initialization for bf16
        # Gate weights are NOT in the state_dict from composition (model had regular Linear layers)
        # We need to add them here so they're properly initialized when loading
        print("\nAdding gate weights to checkpoint...")
        num_experts = len(self.config["experts"])
        gate_count = 0

        for layer_name in list(self.state_dict.keys()):
            # Check if this is a layer that should have a gate
            is_router_layer = any(
                self._is_layer_suitable_for_router(router_layer, layer_name)
                for router_layer in self.config["router_layers"]
            )

            if is_router_layer and '.weight' in layer_name and 'lora' not in layer_name.lower():
                # Extract the base layer name (e.g., "model.layers.0.mlp.gate_proj")
                base_layer_name = layer_name.replace('.weight', '')
                gate_weight_name = f"{base_layer_name}.gate.weight"

                # Get input features from the layer weight shape
                layer_weight = self.state_dict[layer_name]
                in_features = layer_weight.shape[1] if len(layer_weight.shape) > 1 else layer_weight.shape[0]

                # Initialize gate weight with small values for bf16 stability
                gate_weight = torch.zeros(num_experts, in_features, dtype=self.torch_dtype)
                torch.nn.init.normal_(gate_weight, mean=0.0, std=0.01)

                self.state_dict[gate_weight_name] = gate_weight
                gate_count += 1

        print(f"✓ Added {gate_count} gate weight tensors to checkpoint")

        save_pretrained(
            save_directory=checkpoint_path,
            state_dict=self.state_dict,
            tied_weights_keys=self._tied_weights_keys,
            max_shard_size=self.max_shard_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
        tokenizer.save_pretrained(checkpoint_path)
        print(f"checkpoint saved at {checkpoint_path}")
