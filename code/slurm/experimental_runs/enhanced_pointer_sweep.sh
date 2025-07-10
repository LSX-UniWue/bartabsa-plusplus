#!/bin/bash

# Copied over from the thesis experiments

# Base configuration
BASE_CONFIG=""

cd ..

# Function to launch experiment
launch_experiment() {
    local config="$1"
    local name="$2"
    echo "Launching experiment: $name"
    echo "Config: $config"
    sbatch ./sbatch_long_experiment.sh "absa" "$BASE_CONFIG experiment.run_name='$name' $config"
    sleep 1 # Wait a bit between job submissions
}

# Configurations for model variations
declare -A CONFIGS
CONFIGS=(
    ["BART_Baseline"]=""
    ["BART_Baseline_Enhanced"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Roberta2Roberta"]="model.base_model='FacebookAI/roberta-base' model.decoder_model='FacebookAI/roberta-base'"
    ["Roberta2GPT2"]="model.base_model='FacebookAI/roberta-base' model.decoder_model=gpt2"
    ["BartLarge"]="model.base_model=facebook/bart-large"
    ["Roberta2Roberta_Enhanced"]="model.base_model='FacebookAI/roberta-base' model.decoder_model='FacebookAI/roberta-base' model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Roberta2GPT2_Enhanced"]="model.base_model='FacebookAI/roberta-base' model.decoder_model=gpt2 model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["BartLarge_Enhanced"]="model.base_model=facebook/bart-large model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    
    ["Bert2Bert"]="model.base_model=bert-base-uncased model.decoder_model=bert-base-uncased"
    ["Bert2GPT2"]="model.base_model=bert-base-uncased model.decoder_model=gpt2"
    ["RobertaLarge2RobertaLarge"]="model.base_model='FacebookAI/roberta-large' model.decoder_model='FacebookAI/roberta-large'"
    ["RobertaLarge2GPT2Medium"]="model.base_model='FacebookAI/roberta-large' model.decoder_model='openai-community/gpt2-medium'"
    
    ["Bert2Bert_Enhanced"]="model.base_model=bert-base-uncased model.decoder_model=bert-base-uncased model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Bert2GPT2_Enhanced"]="model.base_model=bert-base-uncased model.decoder_model=gpt2 model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["RobertaLarge2RobertaLarge_Enhanced"]="model.base_model='FacebookAI/roberta-large' model.decoder_model='FacebookAI/roberta-large' model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["RobertaLarge2GPT2Medium_Enhanced"]="model.base_model='FacebookAI/roberta-large' model.decoder_model='openai-community/gpt2-medium' model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    
    # GPT-2 models
    ["GPT2GPT-Base"]="model.base_model='openai-community/gpt2' model.decoder_model='openai-community/gpt2'"
    ["GPT2GPT-Medium"]="model.base_model='openai-community/gpt2-medium' model.decoder_model='openai-community/gpt2-medium'"
    ["GPT2GPT-Large"]="model.base_model='openai-community/gpt2-large' model.decoder_model='openai-community/gpt2-large'"
    ["GPT2GPT-XL"]="model.base_model='openai-community/gpt2-xl' model.decoder_model='openai-community/gpt2-xl'"
    
    ["GPT2GPT-Base_Enhanced"]="model.base_model='openai-community/gpt2' model.decoder_model='openai-community/gpt2' model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["GPT2GPT-Medium_Enhanced"]="model.base_model='openai-community/gpt2-medium' model.decoder_model='openai-community/gpt2-medium' model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["GPT2GPT-Large_Enhanced"]="model.base_model='openai-community/gpt2-large' model.decoder_model='openai-community/gpt2-large' model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["GPT2GPT-XL_Enhanced"]="model.base_model='openai-community/gpt2-xl' model.decoder_model='openai-community/gpt2-xl' model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
)

# Dataset fractions to test
FRACTIONS=(1.0)

# Launch experiments for each configuration and fraction
for config_name in "${!CONFIGS[@]}"; do
    for fraction in "${FRACTIONS[@]}"; do
        config="${CONFIGS[$config_name]} dataset.fraction=$fraction"
        name="Seq2Seq_Tests_${config_name}_Fraction_${fraction}"
        launch_experiment "$config" "$name"
    done
done

echo "All experiments launched!"
