
# Installation

```sh
module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mergoo/bin/activate
source /home/mila/m/maryam.hashemzadeh/projects/envs/mergoo_3/bin/activate
pip install mergoo
pip install -r requirements.txt 
deepspeed==0.15.4 



pip install ninja
pip install bitsandbytes==0.44.1
pip install wheel
MAX_JOBS=10 pip install --no-build-isolation flash-attn
pip freeze > requirements.txt
```

# Activate the venv:
```sh
module load python/3.10
module load cuda/12.1.1
source /home/mila/m/maryam.hashemzadeh/projects/envs/mergoo_3/bin/activate
```

# accelerate setup

The config file is stored here:

`/home/mila/m/maryam.hashemzadeh/scratch/cache/huggingface/accelerate/default_config.yaml`

`accelerate env` will print its configuration.

```sh
accelerate launch \
  --deepspeed_config_file "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/conf/zero_stage_2.json" \
  exmp_1.py
```

# Train Experts:
We re using 'datasets/pubmed/pubmed_qa/ori_pqau.json' as expert domain and 'datasets/beavertail/4_cleaned_330k_train_beaverTails_unsafe_drug_abuse_weapons_banned_substance.csv' as unsade expert. 

For fine-tuning we use '/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substance.csv' datastes.


# Adapters and Datasets

Loras are stored in `projects/loras` directory.

Datasets are stored in `MoE-PEFT/datasets` directory for PubmedQAU and BeaverTails/drug abuse. For PKU SafeRLHF_QA, we use the processed datasets in `datasets_processing/pku_saferlhf_processed` directory. For Camel, we use the processed datasets in `datasets_processing/camel_datasets_processed` directory. For LLukas22_fiqa, we use the processed datasets in `datasets_processing/LLukas22_fiqa` directory. For cybersecurity, we use the processed datasets in `datasets_processing/cybersecurity_qa` directory. For mental health, we use the processed datasets in `datasets_processing/mental_health` directory.


MoE models are stored in `projects/data/output` directory.

P;ease note when we use traning data like: `polished_safe_data/train_mix_safe_categories_50samples_clean_no_overlap.json`, the number 50 is number of samples per category. It means it has 4 categories and for each category we have 50 samples. The categories are, `drug_abuse`, `economic_crime`, `cybercrime`, and `psychological_harm`.


# Setup Mt_Bench
```sh
cd mt_bench
module load python/3.10
module load cuda/12.1.1
# pip install -e ".[model_worker,llm_judge]" pip3 install "fschat[model_worker,webui]"
pip install anthropic
pip install openai
source /envs/mt_bench/bin/activate
```


# main files to run

- to train lora: `projects/lora_finetune.py` with `projects/run_lora.sh`
- to evaluate lora: `projects/lora_eval.py` with `projects/lora_eval.sh`

- to traning the routers: `projects/exmp_1.py` with `projects/run_one.sh`
- to generate responses from moe models: `projects/model_eval.py` with `projects/moe_eval.sh`

- to evaluate the responses: `projects/gpt4o_response_relevancy_evaluator.py` with `projects/gpt4o_eval.sh`

```sh
DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed"

# Iterate over all files in the directory
for file in "$DATASET_DIR"/*; do
  if [[ -f "$file" && $(basename "$file") == test_* ]]; then
    echo "$file"
    # Extract the file name without the 'test_' prefix
    file_name=$(basename "$file" | sed 's/^test_//')
    
    # Pass the file as test_data to moe_eval.sh
    sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_L_top2_20_samples_experts/checkpoint-40" "$file" "evals_output/moe_L_top2_20_samples_nodomains_experts_${file_name}"
  fi
done
```




