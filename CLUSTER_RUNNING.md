# Cluster Running Guide

> [!IMPORTANT]
> **For Public Use**: This document contains example configurations that reference specific server setups and user accounts. As with modifications this may still be useful, we chose to keep it here. However, before using these scripts:
> 1. Update all paths, usernames, and server names to match your environment
> 2. Replace email addresses with your own
> 3. Update container registry URLs and API keys
> 4. Ensure you have access to the required computing resources

This readme covers running experiments on both Kubernetes and SLURM clusters.

## Kubernetes Setup

### Building the Docker Image

Before running experiments, you'll need to build and push the Docker image to some registry. Note that you might need to modify the [`Dockerfile`](code/k8s/Dockerfile) to have the correct permissions setup.

```bash
cd bartabsa-reproduce
export NAME=your_registry.com/your_username/bartabsa-reproduce
fastbuildah bud -t $NAME:2.5 -t $NAME:latest -f k8s/Dockerfile .
fastbuildah login your_registry.com
fastbuildah push $NAME:2.5
fastbuildah push $NAME:latest
```

### Running Kubernetes Experiments

> [!NOTE]
> After moving the main experiments to the SLURM cluster, the k8s setup was mostly used for one-off experiments, due to its high flexibility.
> You will probably need to modify the [`run_experiment.py`](code/k8s/run_experiment.py) script to fit your needs, especially the constants at the top of the file.

Use `run_experiment.py` to submit jobs to the Kubernetes cluster:

```bash
python code/k8s/run_experiment.py [options]
```

Options:
- `--run_name`: Custom name for the experiment run
- `--config`: Name of the configuration file to use
- `--gpu_type`: GPU type to use (default: gtx1080ti)
  - Available: a100, rtx4090, rtx2080ti, gtx1080ti, titan, rtx8000
- `--extra_args`: Additional arguments to pass to the run script
- `--clean_runs`: Only clean unused temporary folders
- `--commit`: Commit the run to the repository to log exact state of the code

The script will:
1. Create a temporary folder for the experiment
2. Copy the code to the temporary folder
3. Submit a Kubernetes job with the specified configuration
4. Optionally commit the run to the repository

Example usage:
```bash
# Run with custom configuration
python code/k8s/run_experiment.py --run_name "my_experiment" --config "absa_config.yaml" --gpu_type "rtx4090"

# Clean up unused temporary folders
python code/k8s/run_experiment.py --clean_runs

# Run with extra arguments
python code/k8s/run_experiment.py --extra_args "model.batch_size=32 training.epochs=10"
```


## SLURM Setup

The SLURM cluster is used for running the main experiments. Here's how to get started:

### Environment Setup

1. Cache the Apptainer image (one-time setup):
   ```bash
   sbatch code/slurm/apptainer_caching.sh
   ```

2. For interactive development:
   ```bash
   ./code/slurm/interactive_session.sh [options]
   ```
   Common options:
   - `-g, --gpus`: Number of GPUs (default: 1)
   - `-m, --memory`: Memory allocation (default: 32G)
   - `-t, --time`: Time limit (default: 4:00:00)

### Running Experiments

There are two main ways to run experiments:

1. Single Task Run:
   ```bash
   # For quick experiments (4 hour limit)
   sbatch code/slurm/sbatch_short_experiment.sh <task_name> [config_overrides]
   
   # For longer experiments (16 hour limit)
   sbatch code/slurm/sbatch_long_experiment.sh <task_name> [config_overrides]
   ```

   Example:
   ```bash
   # Run ABSA task with custom batch size
   sbatch code/slurm/sbatch_short_experiment.sh absa "model.batch_size=32"
   ```

2. Parameter Sweeps:
   Create a sweep script in `code/slurm/sweeps/`:

   ```bash
   #!/bin/bash
   
   # Function to launch experiment
   launch_experiment() {
       local config="$1"
       local name="$2"
       sbatch code/slurm/sbatch_short_experiment.sh "absa" "experiment.run_name='$name' $config"
       sleep 1  # Prevent overwhelming scheduler
   }
   
   # Define configurations to test
   declare -A CONFIGS=(
       ["Baseline"]="model.dropout=0.1"
       ["HighDropout"]="model.dropout=0.3"
   )
   
   # Launch experiments
   for config_name in "${!CONFIGS[@]}"; do
       config="${CONFIGS[$config_name]}"
       name="Sweep_${config_name}"
       launch_experiment "$config" "$name"
   done
   ```

Each experiment will automatically:
- Run with 5 different seeds (42, 1337, 1, 123, 420)
- Create a temporary working directory
- Clean up after completion
- Log outputs to `/home/<user>/slurm_logs/`

## Data Structure <a name="data-structure"></a>

### SLURM Setup
The data structure is configured in [`code/conf/slurm_config.yaml`](code/conf/slurm_config.yaml):

```yaml
directories:
  data: /data/absa/pengb/json        # Main data directory
  checkpoints: /tmp/modelcheckpoints  # Model checkpoints
  logs: /tmp/logs                     # Training logs
  predictions: /tmp/predictions       # Model predictions
  heatmaps: /tmp/heatmaps            # Embedding visualizations
```

Ensure your data is organized in this structure before running experiments.

### Kubernetes Setup
For K8s, the data structure is defined in the job templates and expects:
```
/data/
  ├── pengb-json-renamed/    # Main dataset directory
  ├── checkpoints/           # Model checkpoints
  ├── logs/                  # Training logs
  ├── predictions/           # Model predictions
  └── heatmaps/             # Embedding visualizations
```

## Development Workflow

### Interactive Development (SLURM)
The interactive session is designed for direct development work when your local changes are synchronized to the cluster (e.g., using VSCode's Remote-SSH with rsync):

1. Set up local-to-cluster synchronization:
   - Use VSCode's Remote-SSH extension
   - Configure `.vscode/settings.json` for rsync:
   ```json
   {
     "sync-rsync.sites": [
       {
         "name": "cluster",
         "localPath": "path/to/local/project",
         "remotePath": "path/to/cluster/project",
         "hostname": "cluster.hostname",
         "username": "your_username"
       }
     ]
   }
   ```

2. Start an interactive session:
   ```bash
   ./code/slurm/interactive_session.sh --gpus 1 --memory 32G
   ```

3. Your local changes will be immediately available on the cluster for testing

This setup is ideal for debugging model changes, testing new configurations, quick experimental iterations, and development with immediate feedback but not meant for running final experiments.
