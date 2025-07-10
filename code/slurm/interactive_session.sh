#!/bin/bash

# Make sure the script is executable on the cluster!
# chmod +x interactive_session.sh

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -p, --partition   Specify the partition (default: standard)"
    echo "  -g, --gpus        Specify the number of GPUs (default: 1)"
    echo "  -m, --memory      Specify the memory (default: 32G)"
    echo "  -c, --cpus        Specify the number of CPUs (default: 32)"
    echo "  -t, --time        Specify the time limit (default: 4:00:00)"
    echo "  --temp            Specify the temporary space (default: 100G)"
}

# Default values
PARTITION="standard"
GPUS=1
MEMORY="32G"
CPUS=32
TIME="4:00:00"
TEMP="100G"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        -p|--partition) PARTITION="$2"; shift ;;
        -g|--gpus) GPUS="$2"; shift ;;
        -m|--memory) MEMORY="$2"; shift ;;
        -c|--cpus) CPUS="$2"; shift ;;
        -t|--time) TIME="$2"; shift ;;
        --temp) TEMP="$2"; shift ;;
        *) echo "Unknown parameter: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Run the interactive session
srun --pty \
-p $PARTITION \
--gres=gpu:$GPUS \
--mem=$MEMORY \
-c $CPUS \
--time=$TIME \
--tmp=$TEMP \
apptainer shell --nv \
-B /home/your_user/projects/bartabsa-reproduce:/bartabsa-reproduce \
-B /home/your_user/data/bartabsa:/data \
--pwd /bartabsa-reproduce \
--env PATH="/opt/conda/bin:$PATH" \
docker://your-registry/bartabsa-lightning:latest  # Replace with your container registry