#! /bin/bash

# Using the config.yaml file as the base config
BASE_CONFIG=""

cd ..

# Function to launch experiment
launch_experiment() {
    local config="$1"
    local name="$2"
    echo "Launching experiment: $name"
    echo "Config: $config"
    sbatch ./sbatch_absa_experiment.sh "$BASE_CONFIG experiment.run_name='$name' $config"
    sleep 1  # Wait a bit between job submissions
}


# Configurations for subtasks
declare -A CONFIGS
CONFIGS=(
    ["Baseline"]="model.gating_mode=no_gating model.normalize_encoder_outputs=false model.attention_mechanism=none model.use_rms_for_encoder_norm=false"
    ["Baseline_NoMLP"]="model.gating_mode=no_gating model.normalize_encoder_outputs=false model.attention_mechanism=none model.use_rms_for_encoder_norm=false model.use_encoder_mlp=false"
    ["Normalization_Only"]="model.gating_mode=no_gating model.normalize_encoder_outputs=true model.attention_mechanism=none model.use_rms_for_encoder_norm=false"
    ["Full_Enhanced"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false"
    ["Full_Enhanced_Bart_Attention"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=false"
    
    ["Abl_Full_Enhanced_Value_and_DimNorm"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=true model.use_dimension_normalization=true model.use_rms_for_encoder_norm=false"
    ["Abl_Full_Enhanced_RMS_Everywhere"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=true"
    
    ["Full_Enhanced_Fixed"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Abl_Full_Enhanced_RMS_Nowhere_Fixed"]=" model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.dont_use_rms=true model.use_final_layer_norm=true"
    ["Abl_Full_Enhanced_RMS_Everywhere_Fixed"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=true model.use_final_layer_norm=true"
    ["Abl_Full_Enhanced_Value_and_DimNorm_Fixed"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=true model.use_dimension_normalization=true model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Full_Enhanced_Bart_Attention_Fixed"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    
    ["Full_Enhanced_NoGating_NoFinalNorm"]="model.gating_mode=encoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=false"
    ["Full_Enhanced_NoGating_FinalNorm"]="model.gating_mode=encoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=false model.use_dimension_normalization=false model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Full_Enhanced_Bart_Attention_NoGating"]="model.gating_mode=encoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Full_Enhanced_Bart_Attention_NoGating_NoFinalNorm"]="model.gating_mode=encoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=false model.use_final_layer_norm=false"
    
    ["Nearly_Antons_Dream_Architecture"]="model.gating_mode=encoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=true model.use_dimension_normalization=true model.use_rms_for_encoder_norm=false model.use_final_layer_norm=false"
    ["Antons_Dream_Architecture"]="model.gating_mode=encoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_value_matrix=true model.use_dimension_normalization=true model.use_rms_for_encoder_norm=true model.use_final_layer_norm=false"
    
    
    ["Full_Enhanced_Bart_Attention_No_Enc_Gating"]="model.gating_mode=decoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true"
    ["Full_Enhanced_Bart_Attention_RMS_Only"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=true model.use_final_layer_norm=true"
    ["Full_Enhanced_Bart_Attention_No_RMS"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=false model.use_final_layer_norm=true model.dont_use_rms=true"
    
    ["Full_Enhanced_Bart_Attention_No_Enc_Gating_No_Final_Norm"]="model.gating_mode=decoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=false model.use_final_layer_norm=false"
    ["Full_Enhanced_Bart_Attention_No_Enc_Gating_No_Final_Norm_RMS_Enc_Norm"]="model.gating_mode=decoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_rms_for_encoder_norm=true model.use_final_layer_norm=false"
    
    ["Baseline_Large"]="model.gating_mode=no_gating model.normalize_encoder_outputs=false model.attention_mechanism=none model.dont_use_rms=false model.use_final_layer_norm=false model.base_model=facebook/bart-large"
    ["Enc_Norm_Large"]="model.gating_mode=no_gating model.normalize_encoder_outputs=true model.attention_mechanism=none model.dont_use_rms=false model.use_final_layer_norm=false model.base_model=facebook/bart-large"
    ["Full_Enhanced_Model_Large"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.dont_use_rms=false model.use_final_layer_norm=true model.base_model=facebook/bart-large"
    ["Full_Enhanced_Model_Large_No_Enc_Gating"]="model.gating_mode=decoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.dont_use_rms=false model.use_final_layer_norm=true model.base_model=facebook/bart-large"
    ["Full_Enhanced_Model_Large_No_Dec_Gating"]="model.gating_mode=encoder_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.dont_use_rms=false model.use_final_layer_norm=true model.base_model=facebook/bart-large"
    ["Full_Enhanced_Model_Large_No_Final_Norm"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.dont_use_rms=false model.use_final_layer_norm=false model.base_model=facebook/bart-large"
    ["Full_Enhanced_Model_Large_NormalLayerNorm"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.dont_use_rms=true model.use_final_layer_norm=true model.base_model=facebook/bart-large"

    ["Full_Enhanced_Model_Large_No_Attention"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=none model.dont_use_rms=false model.use_final_layer_norm=true model.base_model=facebook/bart-large"
    ["Full_Enhanced_Model_Large_No_EncoderNorm"]="model.gating_mode=full_gating model.normalize_encoder_outputs=false model.attention_mechanism=bart model.dont_use_rms=false model.use_final_layer_norm=true model.base_model=facebook/bart-large"
)

# Launch experiments for each subtask configuration and dataset fraction
for config_name in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$config_name]}"
    name="Architecture_Cleanup_${config_name}"
    launch_experiment "$config" "$name"
done

echo "All experiments launched!"