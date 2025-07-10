#!/bin/bash

task=$1
seed=$2
temp_folder=$3
sif_file=$4
additional_config=${5:-""}
echo "Calling run_absa.sh with task $task, seed $seed, temp_folder $temp_folder, additional_config $additional_config"

# If the additional_config is not empty, make sure a space is added
if [ -n "$additional_config" ]; then
    additional_config=" $additional_config"
fi

data_folder="/home/your_user/data/bartabsa"  # Update path

# Set up Apptainer cache directory
mkdir -p /tmp/apptainer_tmp
export APPTAINER_CACHEDIR="/tmp/apptainer_tmp"
# export APPTAINER_CACHEDIR="/home/your_user/apptainer_tmp/bartabsa"

# Set up Apptainer command
APPTAINER_CMD="apptainer run --env WANDB_API_KEY=$WANDB_API_KEY --nv \
-B $data_folder:/data \
-B $temp_folder:/bartabsa-reproduce \
--pwd /bartabsa-reproduce \
$sif_file"

# Weird CUDA version issues, don't ask...
$APPTAINER_CMD pip uninstall -y flash-attn

# For the absa task we always test all datasets
if [ "$task" == "absa" ]; then
    dataset_names=("14lap" "14res" "15res" "16res")
    # dataset_names=("16res")
    for dataset_name in "${dataset_names[@]}"; do
        $APPTAINER_CMD python src/run.py \
        --config-name $task \
        $additional_config \
        experiment.seed=$seed \
        dataset.name=$dataset_name
        
        # Delete content of the checkpoint folder to keep the temp size manageable
        rm -rf /tmp/modelcheckpoints/*
    done
    # Otherwise we run the task for the given dataset name
else
    $APPTAINER_CMD python src/run.py \
    --config-name $task \
    $additional_config \
    experiment.seed=$seed
fi