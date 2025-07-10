#!/bin/bash
#SBATCH --job-name=Full_English_BARTABSA_Experiments
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --tmp=100G
#SBATCH --time=04:00:00
#SBATCH --output=/home/your_user/slurm_logs/full_slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@example.com  # Replace with your email
#SBATCH --export=ALL,WANDB_API_KEY
#SBATCH --array=1-25  # 5 tasks * 5 seeds

# Define experiment-specific parameters
export TASKS=(absa ssa sre deft spaceeval)
export SEEDS=(42 1337 1 123 420)
export SEED_COUNT=5

echo "Running script $0 with task $1"

# Invoke the unified run script
chmod +x ./common_experiment.sh
source ./common_experiment.sh "$1"