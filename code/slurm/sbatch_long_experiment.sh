#!/bin/bash
#SBATCH --job-name=Full_English_BARTABSA_Experiments
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --tmp=100G
#SBATCH --time=12:00:00
#SBATCH --output=/home/your_user/slurm_logs/full_slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@example.com  # Replace with your email
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --array=1-5  # 5 seeds

# Get the task from the command line argument
TASK=$1
echo "Running script $0 with task $TASK and config $2"

# Define experiment-specific parameters
export TASKS=("$TASK")
export SEEDS=(42 1337 1 123 420)
export SEED_COUNT=5


# Invoke the unified run script
chmod +x ./common_experiment.sh
source ./common_experiment.sh "$2"