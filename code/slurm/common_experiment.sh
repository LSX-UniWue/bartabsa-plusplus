#!/bin/bash

# Parse command-line arguments
additional_config=$1

if [ -z "$additional_config" ]; then
    echo "No additional config options provided, only using defaults."
else
    echo "Additional config: $additional_config"
fi

# Set up environment variables
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# Tasks, seeds and seed count will be set via environment variables when submitting the job

# Validate required environment variables
if [ -z "${TASKS+x}" ] || [ -z "${SEEDS+x}" ] || [ -z "${SEED_COUNT+x}" ]; then
    echo "Error: TASKS, SEEDS, and SEED_COUNT environment variables must be set."
    exit 1
fi

# Calculate task index and seed index
task_index=$(( ($SLURM_ARRAY_TASK_ID - 1) / SEED_COUNT ))
seed_index=$(( ($SLURM_ARRAY_TASK_ID - 1) % SEED_COUNT ))

# Get the task and seed for this job array task
task=${TASKS[$task_index]}
seed=${SEEDS[$seed_index]}

# Create a unique temporary folder for this run
temp_folder="/tmp/bartabsa_run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$temp_folder"

# Copy the sif file to the temp folder
sif_file="/home/your_user/apptainer_images/bartabsa-lightning.sif"  # Update path
temp_sif_file="$temp_folder/bartabsa-lightning.sif"
cp "$sif_file" "$temp_sif_file"

# Copy code to temp folder
# Update this path to match your setup
cp -r /home/your_user/projects/bartabsa-reproduce/* "$temp_folder"

# Change to temp folder
cd "$temp_folder" || exit

echo "Running task $task with seed $seed"
echo "Temp folder: $temp_folder"

# Display current directory and contents
pwd
ls -la

# Run the task script
# Ensure the run_bartabsa.sh is executable
chmod +x slurm/run_bartabsa.sh
./slurm/run_bartabsa.sh "$task" "$seed" "$temp_folder" "$temp_sif_file" "$additional_config"

echo "Finished running task $task with seed $seed"

# Clean up
cd /tmp || exit
rm -rf "$temp_folder"
echo "Cleaned up temp folder $temp_folder"