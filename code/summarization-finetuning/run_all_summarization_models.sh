#!/bin/bash
# Script to submit all model variants to the SLURM cluster

# Set debug mode if provided as argument
debug_flag=${1:-""}  # Empty by default (no debug mode)

echo "Submitting all summarization model variants to the SLURM cluster"
echo "Running with optimized parameters: 3 epochs, faster validation"
if [ "$debug_flag" == "debug" ]; then
    echo "Running in debug mode"
fi

# BERT2BERT models
echo "Submitting BERT2BERT base model..."
sbatch sbatch_summarization.sh bert2bert "$debug_flag"

echo "Submitting BERT2BERT large model..."
sbatch sbatch_summarization.sh bert2bert-large "$debug_flag"

# RoBERTa2RoBERTa models
echo "Submitting RoBERTa2RoBERTa base model..."
sbatch sbatch_summarization.sh roberta2roberta "$debug_flag"

echo "Submitting RoBERTa2RoBERTa large model..."
sbatch sbatch_summarization.sh roberta2roberta-large "$debug_flag"

# GPT2-to-GPT2 models (different sizes)
echo "Submitting GPT2-to-GPT2 base model..."
sbatch sbatch_summarization.sh gpt22gpt2 "$debug_flag"

echo "Submitting GPT2-to-GPT2 medium model..."
sbatch sbatch_summarization.sh gpt22gpt2-medium "$debug_flag"

echo "Submitting GPT2-to-GPT2 large model..."
sbatch sbatch_summarization.sh gpt22gpt2-large "$debug_flag"

# BART models
echo "Submitting BART base model..."
sbatch sbatch_summarization.sh bart "$debug_flag"

echo "Submitting BART large model..."
sbatch sbatch_summarization.sh bart-large "$debug_flag"

echo "All jobs submitted!"
echo "Monitor your jobs with: squeue -u $USER"