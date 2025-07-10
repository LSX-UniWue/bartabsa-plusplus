#!/bin/bash

# Run the most promising versions of the models for all tasks (not just ABSA)

# Base configuration
BASE_CONFIG=""

cd ..

# Function to launch experiment
launch_experiment() {
    local task="$1"
    local config="$2"
    local name="$3"
    echo "Launching experiment: $name"
    echo "Config: $config"
    
    sbatch ./sbatch_long_experiment.sh "$task" "$BASE_CONFIG experiment.run_name='$name' $config"
    sleep 1  # Wait a bit between job submissions
}

# Tasks
TASKS=(ssa sre deft)

# Configurations for model variations
declare -A CONFIGS
CONFIGS=(
    ["Baseline_Large"]="model.gating_mode=no_gating model.normalize_encoder_outputs=false model.attention_mechanism=none model.dont_use_rms=false model.use_final_layer_norm=false model.base_model=facebook/bart-large"
    ["Large_Enhanced"]="model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.dont_use_rms=false model.use_final_layer_norm=true model.base_model=facebook/bart-large"
)

# Launch experiments for each task
for task in "${TASKS[@]}"; do
    for config_name in "${!CONFIGS[@]}"; do
        config="${CONFIGS[$config_name]} dataset.task=$task"
        name="${task}_Model_Updates_${config_name}"
        launch_experiment "$task" "$config" "$name"
    done
done

echo "All experiments launched!"