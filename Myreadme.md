
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



