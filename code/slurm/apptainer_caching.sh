#!/bin/bash
#SBATCH --job-name=Cache_BARTABSA_Image
#SBATCH --comment="Simple helper script only used to cache the apptainer image on once on the hpc, since all subsequent jobs can run in parallel and will use the cached image."
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -p small_cpu
#SBATCH --output=/home/s392248/slurm_logs/cache_apptainer_image-%j.out

# Set up Apptainer cache directory
export APPTAINER_CACHEDIR="/home/s392248/apptainer_tmp/bartabsa"

# Pull the image
apptainer pull docker://your-registry/bartabsa-lightning:latest  # Replace with your container registry

echo "Apptainer image cached successfully"
