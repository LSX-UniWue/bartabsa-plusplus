#!/bin/bash
#SBATCH --job-name=Summarization_Finetuning
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --tmp=100G
#SBATCH --time=23:00:00
#SBATCH --output=/home/your_user/slurm_logs/summarization-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@example.com  # Replace with your email
#SBATCH --export=ALL,WANDB_API_KEY,HUGGINGFACE_TOKEN

# Parse command-line arguments
model_type=${1:-"bert2bert"}  # Default to bert2bert if not specified
debug_flag=${2:-""}  # Empty by default (no debug mode)
seed=${3:-42}  # Default seed is 42 if not specified

# Set debug flag if requested
if [ "$debug_flag" == "debug" ]; then
    debug_arg="--debug_thingy"
    echo "Running in debug mode"
else
    debug_arg=""
fi

# Create a unique temporary folder for this run
temp_folder="/tmp/summarization_run_${SLURM_JOB_ID}"
mkdir -p "$temp_folder"

# Copy the sif file to the temp folder
sif_file="/home/your_user/apptainer_images/bartabsa-lightning.sif"  # Update path
temp_sif_file="$temp_folder/bartabsa-lightning.sif"
cp "$sif_file" "$temp_sif_file"

# Copy code to temp folder
cp -r /home/your_user/projects/bartabsa-reproduce/* "$temp_folder"  # Update path

# Change to temp folder
cd "$temp_folder" || exit

echo "Running summarization fine-tuning with model type $model_type and seed $seed"
echo "Temp folder: $temp_folder"

# Display current directory and contents
pwd
ls -la

# Set up Apptainer cache directory
mkdir -p /tmp/apptainer_tmp
export APPTAINER_CACHEDIR="/tmp/apptainer_tmp"

# Set up Apptainer command
APPTAINER_CMD="apptainer run --env WANDB_API_KEY=$WANDB_API_KEY --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN --nv \
--pwd /bartabsa-reproduce \
-B $temp_folder:/bartabsa-reproduce \
$temp_sif_file"

# Determine model name based on model type
case "$model_type" in
    "bert2bert")
        model_name="bert-base-uncased"
    ;;
    "bert2bert-large")
        model_name="bert-large-uncased"
        model_type="bert2bert"
    ;;
    "roberta2roberta")
        model_name="roberta-base"
    ;;
    "roberta2roberta-large")
        model_name="roberta-large"
        model_type="roberta2roberta"
    ;;
    "gpt22gpt2")
        model_name="gpt2"
    ;;
    "gpt22gpt2-medium")
        model_name="gpt2-medium"
        model_type="gpt22gpt2"
    ;;
    "gpt22gpt2-large")
        model_name="gpt2-large"
        model_type="gpt22gpt2"
    ;;
    "bart")
        model_name="facebook/bart-base"
    ;;
    "bart-large")
        model_name="facebook/bart-large"
        model_type="bart"
    ;;
    *)
        echo "Unknown model type: $model_type. Using bert2bert as default."
        model_type="bert2bert"
        model_name="bert-base-uncased"
    ;;
esac

# Install evaluate and rouge-score and accelerate
$APPTAINER_CMD pip install evaluate rouge-score accelerate

# Set error handling
set -e  # Exit immediately if a command exits with a non-zero status
trap 'echo "Error occurred. Cleaning up..."; cd /tmp || exit; rm -rf "$temp_folder"; exit 1' ERR

# Set learning rate based on model type
if [[ "$model_type" == "bart" ]]; then
    learning_rate="2e-5"  # Lower learning rate for BART models
    echo "Using lower learning rate for BART model: $learning_rate"
else
    learning_rate="5e-5"  # Original learning rate for other models
    echo "Using standard learning rate: $learning_rate"
fi

# Run the Python script
echo "Starting training with model type $model_type at $(date)"
$APPTAINER_CMD python summarization-finetuning/finetune_summarization.py \
--model_type $model_type \
--model_name_or_path $model_name \
--output_dir /bartabsa-reproduce/outputs/${model_type}_${seed} \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate $learning_rate \
--num_train_epochs 3 \
--logging_steps 500 \
--evaluation_strategy steps \
--eval_steps 2000 \
--save_strategy steps \
--save_steps 2000 \
--warmup_steps 1000 \
--max_train_samples 100000 \
--max_eval_samples 1000 \
--max_test_samples 1000 \
--preprocessing_num_workers 16 \
--metric_for_best_model rouge2 \
--greater_is_better true \
--seed $seed \
--bf16 \
--push_to_hub \
--predict_with_generate \
        --hub_model_id "your_huggingface_username/${model_type}-${model_name##*/}-cnn-dailymail-seed${seed}" \  # Replace with your HuggingFace username
--run_name "julia-${model_type}-${model_name##*/}-${seed}" \
--hub_private_repo \
$debug_arg

# Check if the training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
else
    echo "Training failed with exit code $? at $(date)"
    exit 1
fi

echo "Finished running summarization fine-tuning with model type $model_type and seed $seed"

# Clean up
cd /tmp || exit
rm -rf "$temp_folder"
echo "Cleaned up temp folder $temp_folder"
pwd
ls -la

