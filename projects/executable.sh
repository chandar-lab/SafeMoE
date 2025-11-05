#task_nums=(22) #(22 21)

###################################### run for shuffle label
# ranks=(32)
# epochs=(8)


# for r in ${ranks[@]}
# do
#   for epoch in ${epochs[@]}
#   do
#     echo $r $epoch
#     # output_dir_1="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/checkpoint/"
#     # sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_test.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_train.json" $output_dir_1 False
#     output_dir_1="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/checkpoint_phase2/"
#     sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_test_phase2.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/downsampled_prm800k_train_phase2.json" $output_dir_1 False
#     # output_dir_2="/home/mila/m/maryam.hashemzadeh/scratch/verifier/gp_verifier/finalanswer_checkpoint_phase2/"
#     # sbatch run_lora.sh $r $epoch "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_test_phase2.json" "/home/mila/m/maryam.hashemzadeh/scratch/verifier/finalanswer_downsampled_prm800k_train_phase2.json" $output_dir_2 True
#   done
# done

# sbatch run.sh 'QuantFactory/Llama-3-8B-Instruct-Finance-RAG-GGUF' ### tokenizer is not found 
# sbatch run.sh 'HPAI-BSC/Llama3-Aloe-8B-Alpha'
# sbatch run.sh 'AdaptLLM/medicine-LLM-13B'
# sbatch run.sh 'camel-ai/CAMEL-13B-Combined-Data'
# sbatch run.sh 'meta-llama/Llama-3.2-3B-Instruct'
# sbatch run.sh 'meta-llama/Llama-3.2-3B'

# sbatch run.sh 'meta-llama/Llama-3.1-8B'
# sbatch run.sh 'meta-llama/Llama-3.1-8B-Instruct'
# sbatch run.sh 'meta-llama/Meta-Llama-3-8B'

# sbatch run.sh 'meta-llama/Llama-2-13b-hf'
# sbatch run.sh 'lmsys/vicuna-13b-v1.5'
# sbatch run.sh 'meta-llama/Llama-3.2-1B'
# sbatch run.sh 'meta-llama/Llama-3.2-1B-Instruct'

# sbatch run.sh 'meta-llama/Llama-2-7b-chat-hf' 'fwnlp'
# sbatch run.sh 'meta-llama/Llama-2-7b-chat-hf' 'llm_attack'


# tasks=("a-safety" "adversarial_qa_droberta_answer_the_following_q" "camel-biology" "beaverTails_unsafe_animal_abuse" "beaverTails_unsafe_child_abuse" "beaverTails_unsafe_controversial_topics_politics" "beaverTails_unsafe_discrimination_stereotype_injustice" "beaverTails_unsafe_drug_abuse_weapons_banned_substance" "beaverTails_unsafe_financial_crime_property_crime_theft" "beaverTails_unsafe_hate_speech_offensive_language" "beaverTails_unsafe_misinformation_regarding_ethics_laws_and_safety" "beaverTails_unsafe_non_violent_unethical_behavior" "beaverTails_unsafe_privacy_violation" "beaverTails_unsafe_self_harm" "beaverTails_unsafe_sexually_explicit_adult_content" "beaverTails_unsafe_terrorism_organized_crime_unsafe" "beaverTails_unsafe_violence_aiding_and_abetting_incitement")

##################################################

sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_unsafe_drug_abuse_weapons_banned_substance.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/Qwen2.5-1.5B-lora-drug_abuse/"

sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/pubmed/pubmed_qa/ori_pqau_instruction_response_10k_sample.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/Qwen2.5-1.5B-lora-pubmedqau/"

###################################################
# TRAIN LORA
###################################################
### PKU safetyRLHF

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_psychological_harm.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_psychological/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_physical_harm.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_physical/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_environmental_damage.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_environmental/"

sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_cybercrime.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/Qwen2.5-1.5B-lora-unsafe_cybercrime/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_economic_crime.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_economic/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substance.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-safe_drug_abuse_weapons_banned_substance/"

### camel
# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/camel_datasets_processed/biology_train.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-biology_camel/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/camel_datasets_processed/chemistry_train.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-chemistry_camel/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/camel_datasets_processed/physics_train.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-physics_camel/"

### others
# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/LLukas22_fiqa/fiqa_train.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-fiqa/"

sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/cybersecurity_qa/cybersecurity_qa_train.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/Qwen2.5-1.5B-lora-cybersecurity_kaggle/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/mental_health/Amod_mental_health_counseling_conversations_train.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-mental_health/"

### other unsafe
# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_copyright_issues.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_copyright_issues/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_discriminatory_behavior.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_discriminatory_behavior/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_disrupting_public_order.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_disrupting_public_order/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_endangering_national_security.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_endangering_national_security/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_endangering_public_health.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_endangering_public_health/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_human_trafficking.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_human_trafficking/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_insulting_behavior.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_insulting_behavior/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_mental_manipulation.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_mental_manipulation/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_privacy_violation.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_privacy_violation/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_sexual_content.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_sexual_content/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_violence.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_violence/"

# sbatch run_lora.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/train_unsafe_white_collar_crime.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-unsafe_white_collar_crime/"

### safe adapters


###################################################
# EVALUATE LORA
###################################################


# sbatch lora_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substance.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-safe_drug_abuse_weapons_banned_substance/" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/adapter_safe_drug_abuse.json"
# sbatch lora_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/4_cleaned_330k_train_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substance.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-pubmedqau/" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/adapter_pubmedqau.json"

# sbatch lora_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/2_cleaned_330k_test_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substanc.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-safe_drug_abuse_weapons_banned_substance/" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/adapter_safe_drug_abuse.json"
# sbatch lora_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/2_cleaned_330k_test_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substanc.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-pubmedqau/" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/adapter_pubmedqau.json"


# sbatch lora_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/mental_health/Amod_mental_health_counseling_conversations_test.json" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/loras/mistral-lora-mental_health" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output/adapter_mental_health.json"


###################################################
# EVALUATE MOE
###################################################

## drug abuse
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_drug_abuse_pubmedqau_top1/checkpoint-3910" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/2_cleaned_330k_test_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substanc.json" "evals_output/moe_model_drugabuse_top1.json"
# all moe
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/2_cleaned_330k_test_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substanc.json" "evals_output/moe_model_all_drugabuse_top2.json"

## cybersecurity
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_cybersecurity_top1/checkpoint-3640" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_cybercrime.json" "evals_output/moe_model_cybercrime_top1.json"
# all moe
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_cybercrime.json" "evals_output/moe_model_all_cybercrime_top2.json"

# ## economic
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_economic_top1/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_economic_crime.json" "evals_output/moe_model_economic_crime_top1.json"
# # all moe
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_economic_crime.json" "evals_output/moe_model_all_economic_crime_top2.json"

# ## mental health
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_mental_health_top1/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/polished_safe_data/test_safe_gpt4_qtype4_psychological_harm_polished.json" "evals_output/moe_model_health_top1.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_mental_health_2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/polished_safe_data/test_safe_gpt4_qtype4_psychological_harm_polished.json" "evals_output/moe_model_health_top2.json"
# # all moe
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/polished_safe_data/test_safe_gpt4_qtype4_psychological_harm_polished.json" "evals_output/moe_model_all_health_top2.json"

###################################################
# EVALUATE MOE L and XL
###################################################

# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/polished_safe_data/test_safe_gpt4_qtype4_psychological_harm_polished.json" "evals_output/moe_model_all_L_14apater_top2_health_top2.json"

# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/polished_safe_data/test_safe_gpt4_qtype4_psychological_harm_polished.json" "evals_output/moe_model_all_XL_22apater_top2_health_top2.json"


# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_copyright_issues.json" "evals_output/moe_model_all_L_14apater_top2_unsafe_copyright_issues.json"

# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_copyright_issues.json" "evals_output/moe_model_all_XL_22apater_top2_unsafe_copyright_issues.json"


# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_animal_abuse.json" "evals_output/moe_model_all_L_14apater_top2_unsafe_animal_abuse.json"

# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_animal_abuse.json" "evals_output/moe_model_all_XL_22apater_top2_unsafe_animal_abuse.json"

# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_discriminatory_behavior.json" "evals_output/moe_model_all_L_14apater_top2_unsafe_discriminatory_behavior.json"

# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed/test_unsafe_discriminatory_behavior.json" "evals_output/moe_model_all_XL_22apater_top2_unsafe_discriminatory_behavior.json"



#!/bin/bash

# # Directory containing the files
# DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed"

# # Iterate over all files in the directory
# for file in "$DATASET_DIR"/*; do
# #   if [[ -f "$file" ]]; then
#   if [[ -f "$file" && $(basename "$file") == test_* ]]; then
#     echo "$file"
#     # Pass the file as test_data to moe_eval.sh
#     # sbatch moe_eval.sh --test_data "$file"
#     sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "$file" "evals_output/moe_model_all_L_14apater_top2_$file"
#     sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "$file" "evals_output/moe_model_all_XL_22apater_top2_$file"
#   fi
# done

# for job_id in {7468423..7468462}; do
#   # Cancel each job
#   scancel "$job_id"
# done


#!/bin/bash

# # Directory containing the files
# DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed"

# # Iterate over all files in the directory
# for file in "$DATASET_DIR"/*; do
#   if [[ -f "$file" && $(basename "$file") == test_* ]]; then
#     echo "$file"
#     # Extract the file name without the 'test_' prefix
#     file_name=$(basename "$file" | sed 's/^test_//')
    
#     # Pass the file as test_data to moe_eval.sh
#     sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "$file" "evals_output/moe_model_all_L_14apater_top2_${file_name}"
#     sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "$file" "evals_output/moe_model_all_XL_22apater_top2_${file_name}"
#   fi
# done


# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/2_cleaned_330k_test_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substanc.json" "evals_output/moe_model_all_L_14apater_top2_drugabuse.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "/home/mila/m/maryam.hashemzadeh/scratch/saftly/MoE-PEFT/datasets/beavertail/2_cleaned_330k_test_beaverTails_safe_gpt4_qtype4_drug_abuse_weapons_banned_substanc.json" "evals_output/moe_model_all_XL_22apater_top2_drugabuse.json"

# DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output"

# for file in "$DATASET_DIR"/moe_model_all_L_14apater_top2_*; do
#   if [[ -f "$file" ]]; then
#     echo "$file"
#     # Extract the file name without the 'moe_model_all_L_14apater_top2_' prefix
#     file_name=$(basename "$file" | sed 's/^moe_model_all_L_14apater_top2_//')
    
#     # Pass the file as test_data to moe_eval.sh
#     sbatch safety_eval.sh "$file" "evals_output/percentage_moe_model_all_L_14apater_top2_${file_name}"
#   fi
# done

# for file in "$DATASET_DIR"/moe_model_all_XL_22apater_top2_*; do
#   if [[ -f "$file" ]]; then
#     echo "$file"
#     # Extract the file name without the 'moe_model_all_L_14apater_top2_' prefix
#     file_name=$(basename "$file" | sed 's/^moe_model_all_XL_22apater_top2_//')
    
#     # Pass the file as test_data to moe_eval.sh
#     sbatch safety_eval.sh "$file" "evals_output/percentage_moe_model_all_XL_22apater_top2_${file_name}"
#   fi
# done

###################################################
# EVALUATE baseline models
###################################################
## for generatting text
###################################################
DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/pku_saferlhf_processed"

# Iterate over all files in the directory
for file in "$DATASET_DIR"/*; do
  if [[ -f "$file" && $(basename "$file") == test_* ]]; then
    echo "$file"
    # Extract the file name without the 'test_' prefix
    file_name=$(basename "$file" | sed 's/^test_//')
    
    # Pass the file as test_data to moe_eval.sh
    sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_L_top2_20_samples_experts/checkpoint-40" "$file" "evals_output/moe_L_top2_20_samples_nodomains_experts_${file_name}"
    sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_L_top2_50_samples_experts/checkpoint-110" "$file" "evals_output/moe_L_top2_50_samples_nodomains_experts_${file_name}"
    sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_L_top2_100_samples_experts/checkpoint-220" "$file" "evals_output/moe_L_top2_100_samples_nodomains_experts_${file_name}"
    sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_8adapters_top2_20_samples_experts/checkpoint-40" "$file" "evals_output/moe_8adapters_top2_20_samples_nodomains_experts_${file_name}"
    sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_8adapters_top2_50_samples_experts/checkpoint-110" "$file" "evals_output/moe_8adapters_top2_50_samples_nodomains_experts_${file_name}"
    sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_8adapters_top2_100_samples_experts/checkpoint-220" "$file" "evals_output/moe_8adapters_top2_100_samples_nodomains_experts_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_XL_top2_100_samples_nodomains_experts/checkpoint-220" "$file" "evals_output/moe_XL_top2_100_samples_nodomains_experts_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_XL_top2_20_samples/checkpoint-40" "$file" "evals_output/moe_XL_top2_20samples_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_XL_top2_100_2samples/checkpoint-220" "$file" "evals_output/moe_XL_top2_100samples_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_XL_top2_50samples/checkpoint-110" "$file" "evals_output/moe_XL_top2_50samples_${file_name}" 
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_XL_top2_200samples/checkpoint-450" "$file" "evals_output/moe_XL_top2_200samples_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250" "$file" "evals_output/moe_model_all_8apater_top2_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_4domainexperts_top2_top2/checkpoint-1250" "$file" "evals_output/moe_model_4domainexperts_top2_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_2/checkpoint-310" "$file" "evals_output/mistral_sft_model_all_safe_data500sample_2_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_3_200sample/checkpoint-120" "$file" "evals_output/mistral_sft_model_all_safe_data200sample_3_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_3_20sample/checkpoint-40" "$file" "evals_output/mistral_sft_model_all_safe_data20sample_4_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/data/output/mistral_sft_all_safe_data_5_20sample_nooverlap/checkpoint-180" "$file" "evals_output/mistral_sft_model_all_safe_5_20sample_nooverlap_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_8_50sample_nooverlap/checkpoint-110" "$file" "evals_output/mistral_sft_model_all_safe_8_50sample_nooverlap_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_9_100sample_nooverlap/checkpoint-220" "$file" "evals_output/mistral_sft_model_all_safe_9_100sample_nooverlap_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_9_100sample_nooverlap_30epoch_lr-3/checkpoint-660" "$file" "evals_output/mistral_sft_model_all_safe_9_100sample_overlap_30epoch_lr-3_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_10_500sample_nooverlap_30epoch/checkpoint-3360" "$file" "evals_output/mistral_sft_model_all_safe_10_500sample_overlap_30epoch_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_10_500sample_nooverlap_10epoch/checkpoint-1120" "$file" "evals_output/mistral_sft_model_all_safe_10_500sample_overlap_10epoch_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_11_200sample_nooverlap_10epoch/checkpoint-450" "$file" "evals_output/mistral_sft_model_all_safe_11_200sample_overlap_10epoch_${file_name}"
    # sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_11_200sample_nooverlap_30epoch/checkpoint-1350" "$file" "evals_output/mistral_sft_model_all_safe_11_200sample_overlap_30epoch_${file_name}"
  fi
done


# file='/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/advbench_processed/advbench_data.json'
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" $file "evals_output/moe_model_all_L_14apater_top2_advbench_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" $file "evals_output/moe_model_all_XL_22apater_top2_advbench_data_more_2.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_4domainexperts_top2_top2/checkpoint-1250" $file "evals_output/moe_model_4domainexperts_top2_advbench_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_10_500sample_nooverlap_10epoch/checkpoint-1120" $file "evals_output/mistral_sft_model_all_safe_10_500sample_overlap_10epoch_advbench_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_11_200sample_nooverlap_10epoch/checkpoint-450" $file "evals_output/mistral_sft_model_all_safe_11_200sample_overlap_10epoch_advbench_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250" $file "evals_output/moe_model_all_8apater_top2_advbench_data.json"


# file='/home/mila/m/maryam.hashemzadeh/scratch/saftly/datasets_processing/harmbench_processed/harmbench_contextual_data.json'
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" $file "evals_output/moe_model_all_L_14apater_top2_harmbench_contextual_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" $file "evals_output/moe_model_all_XL_22apater_top2_harmbench_contextual_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_4domainexperts_top2_top2/checkpoint-1250" $file "evals_output/moe_model_4domainexperts_top2_harmbench_contextual_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_10_500sample_nooverlap_10epoch/checkpoint-1120" $file "evals_output/mistral_sft_model_all_safe_10_500sample_overlap_10epoch_advbench_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_11_200sample_nooverlap_10epoch/checkpoint-450" $file "evals_output/mistral_sft_model_all_safe_11_200sample_overlap_10epoch_advbench_data.json"
# sbatch moe_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_top2/checkpoint-1250" $file "evals_output/moe_model_all_8apater_top2_harmbench_contextual_data.json"



# sbatch run_vllm_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "moe_all_L_14apater_top2" $file "evals_output/moe_model_all_L_14apater_top2_advbench_data.json"
# sbatch run_vllm_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "moe_all_XL_22apater_top2" "$file" "evals_output/moe_model_all_XL_22apater_top2_advbench_data.json"
# sbatch run_vllm_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_4domainexperts_top2_top2/checkpoint-1250" "moe_4domainexperts_top2_top2" "$file" "evals_output/moe_model_4domainexperts_top2_advbench_data.json"
# sbatch run_vllm_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_10_500sample_nooverlap_10epoch/checkpoint-1120" "sft_500sample_10epoch" "$file" "evals_output/mistral_sft_model_all_safe_10_500sample_overlap_10epoch_advbench_data.json"
# sbatch run_vllm_eval.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_sft_all_safe_data_11_200sample_nooverlap_10epoch/checkpoint-450" "sft_200sample_10epoch" "$file" "evals_output/mistral_sft_model_all_safe_11_200sample_overlap_10epoch_advbench_data.json"

# sbatch run.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_L_14apater_top2/checkpoint-1250" "moe_all_L_14apater_top2"
# sbatch run.sh "/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/data/output/mistral_lora_moe_all_XL_22apater_top2/checkpoint-1250" "moe_all_XL_22apater_top2"

###################################################
## for safety eval
###################################################
# DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output"

# for file in "$DATASET_DIR"/mistral_sft_model_all_safe_10_500sample_overlap_10epoch_*; do
#   if [[ -f "$file" ]]; then
#     echo "$file"
#     # Extract the file name without the 'moe_model_all_L_14apater_top2_' prefix
#     file_name=$(basename "$file" | sed 's/^mistral_sft_model_all_safe_10_500sample_overlap_10epoch_//')
    
#     # Pass the file as test_data to moe_eval.sh
#     sbatch safety_eval.sh "$file" "evals_output/percentage_mistral_sft_model_all_safe_10_500sample_overlap_10epoch_${file_name}"
#   fi
# done



# for job_id in {7632308..7632312}; do
#   # Cancel each job
#   scancel "$job_id"
# done

###################################################
## JPT4o safety and informative eval
###################################################
# DATASET_DIR="/home/mila/m/maryam.hashemzadeh/scratch/saftly/mergoo/projects/evals_output"

# for file in "$DATASET_DIR"/mistral_sft_model_all_safe_10_500sample_overlap_10epoch_*; do
#   if [[ -f "$file" ]]; then
#     echo "$file"
#     # Extract the file name without the 'moe_model_all_L_14apater_top2_' prefix
#     file_name=$(basename "$file" | sed 's/^mistral_sft_model_all_safe_10_500sample_overlap_10epoch_//')
    
#     # Pass the file as test_data to moe_eval.sh
#     sbatch safety_eval.sh "$file" "evals_output/percentage_mistral_sft_model_all_safe_10_500sample_overlap_10epoch_${file_name}"
#   fi
# done


