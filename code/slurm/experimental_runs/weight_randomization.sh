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
    # If large model, use long experiment
    if [[ "$config" == *"large"* ]]; then
        sbatch ./sbatch_long_experiment.sh "absa" "$BASE_CONFIG experiment.run_name='$name' $config"
    else
        sbatch ./sbatch_short_experiment.sh "absa" "$BASE_CONFIG experiment.run_name='$name' $config"
    fi
    sleep 1 # Wait a bit between job submissions
}

# Configurations for model variations
declare -A CONFIGS
CONFIGS=(
    ["Baseline"]="model.randomize_encoder=false model.randomize_decoder=false"
    ["Randomize_Encoder"]="model.randomize_encoder=true model.randomize_decoder=false"
    ["Randomize_Decoder"]="model.randomize_encoder=false model.randomize_decoder=true"
    ["Randomize_Both"]="model.randomize_encoder=true model.randomize_decoder=true"
)
# Dataset fractions to test
FRACTIONS=(1.0)

# Launch experiments for each configuration and fraction
for config_name in "${!CONFIGS[@]}"; do
    for fraction in "${FRACTIONS[@]}"; do
        config="${CONFIGS[$config_name]} dataset.fraction=$fraction"
        name="Weight_Randomization_${config_name}_Fraction_${fraction}"
        launch_experiment "$config" "$name"
    done
done

echo "All experiments launched!"

