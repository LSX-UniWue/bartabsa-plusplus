# BartABSA++: Enhanced Pointer Networks for Aspect-Based Sentiment Analysis

> [!NOTE]
> This repository contains the official implementation of **BartABSA++**, our XLLM@ACL 2025 paper that revisits the BartABSA framework, achieves state-of-the-art results on aspect-based sentiment analysis and experiments with bridging the gap between modern decoder-only LLMs and encoder-decoder pointer networks.

## About This Implementation

This is a **complete from-scratch reimplementation** of the [original BART-ABSA work](https://github.com/yhcc/BARTABSA) using modern libraries ([PyTorch Lightning](https://github.com/Lightning-AI/lightning), updated Transformers) with significant enhancements for modularity, extensibility, and multi-task support. Our contributions include:

1. **Enhanced Architecture**: Improved pointer networks with:
   - Feature normalization for training stability with larger models
   - Parametrized gating mechanisms replacing static hyperparameters
   - Additional cross-attention mechanisms reusing pretrained weights

2. **Multi-Architecture Support**: Extended beyond BART to support encoder-decoder combinations following [Rothe et al. (2020)](https://doi.org/10.1162/tacl_a_00313):
   - BERT, RoBERTa, GPT-2 combinations  
   - Scaling experiments up to 3.6B parameters
   - Systematic evaluation of encoder vs decoder contributions

3. **Multi-Task Framework**: Support for seven different structured prediction tasks beyond ABSA
4. **Modern Implementation**: Updated codebase with better reproducibility, logging, and scalability
5. **Comprehensive Evaluation**: Extensive experiments showing structured approaches remain competitive with modern LLMs

> [!IMPORTANT]
> **Key Finding from Our Work**: Our experiments demonstrate that structured approaches like pointer networks remain highly competitive with modern LLMs for tasks requiring precise relational information extraction. The quality of token-level representations (encoder) is far more important than generative capabilities (decoder) for these structured prediction tasks.

## Key Features

This implementation offers enhanced experimental capabilities, including:
- Comprehensive logging and metrics tracking via [Weights & Biases](https://wandb.ai/site)
- Parameter heatmap visualizations (enable with `experiment.write_heatmaps`)
- Prediction output in multiple formats including JSON and XMI (enable with `experiment.write_predictions`)
- Multi-architecture support (BART, BERT, RoBERTa, GPT combinations)
- Cluster deployment support for both [Kubernetes](https://kubernetes.io/) and [SLURM](https://slurm.schedmd.com/)

## Supported Tasks

This implementation supports the following structured prediction tasks:

1. **ABSA** (Aspect-based Sentiment Analysis) - Main focus, includes datasets: 14lap, 14res, 15res, 16res
2. **SSA** (Structured Sentiment Analysis)  
3. **SRE** (Sentiment Relationship Extraction)
4. **DEFT** (Definition Extraction from Free Text)
5. **SpaceEval** (Space Evaluation)
6. **GABSA** (German Aspect-based Sentiment Analysis)
7. **GNER** (German Named Entity Recognition)

Each task has its own dataset structure and configuration. For more details on the data structure, refer to the [data README](data/README.md).

## Setup

When running the code locally, you should be able to install all necessary dependencies in a virtual environment using the following commands:
```bash
cd code
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```


## Quick Start

### Required Updates
As this is an anonymized version of the code, you will need to make several updates to the codebase before everything works. (only update whats needed for your use case)
- [ ] Most importantly: **Weights & Biases**: Update `entity` in `code/conf/config.yaml` with your W&B team name, otherwise only offline logging will work.

For remote running, you will need to update the following:
- [ ] **Email notifications**: Replace email addresses in SLURM scripts (`code/slurm/sbatch_*.sh`) 
- [ ] **Container registry**: Update Docker/Apptainer image URLs in cluster scripts
- [ ] **File paths**: Update all `/home/your_user/` paths to match your system
- [ ] **Cluster configuration**: Update server names, namespaces, and resource specifications

### Basic Training
```bash
cd code
source venv/bin/activate

# Train BartABSA++ on ABSA (default)
python src/run.py 

# Train on different tasks
python src/run.py --config-name ssa.yaml  # Structured Sentiment Analysis
python src/run.py --config-name deft.yaml # Definition Extraction

# Override specific parameters
python src/run.py model.use_enhanced_architecture=true dataset.name='14res'
```

### ⚙️ Key Configuration Options

Configurations are done using [Hydra](https://hydra.cc/docs/intro) and stored in the [`/code/conf/`](/code/conf/) directory. Some of the most important options are:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.use_enhanced_architecture` | Enable our architectural improvements | `true` |
| `model.encoder_name` | Backbone encoder model | `facebook/bart-base` |
| `dataset.name` | Dataset (14lap, 14res, 15res, 16res) | `14lap` |
| `training.max_epochs` | Maximum training epochs | `200` |
| `training.batch_size` | Batch size | `16` |
| `training.learning_rate` | Learning rate | `5e-5` |

> [!NOTE]
> See [`/code/conf/config.yaml`](/code/conf/config.yaml) for all configuration options. Each task has its own config file in `/code/conf/`.

## Code Overview

The codebase (`code/`) is structured as follows:

- `src/`: Main source code directory
  - `run.py`: Entry point running a single experiment including testing
  - `model/`: Contains the BartABSA++ model implementation with our enhancements
  - `dataset/`: Data loading and preprocessing modules for all tasks
  - `metrics/`: Task-specific evaluation metrics
  - `utils/`: Utility functions and helper classes
- `conf/`: Hydra config files for running experiments
- `k8s/`: Kubernetes-related files for cluster deployment
- `slurm/`: SLURM-related files for cluster deployment

Key components:
- [`model.py`](code/src/model/model.py): The main model class implementing our enhanced architecture
- [`module.py`](code/src/model/module.py): The PyTorch Lightning module for the model
- [`mapping_tokenizer.py`](code/src/dataset/mapping_tokenizer.py): The label conversion for generating the pointer labels

## Training

The training script can be easily configured using [Hydra](https://hydra.cc/docs/intro) via the config files in the `conf/` directory or by directly passing parameters to the script. By default the config file `config.yaml` is used, which works for the ABSA task on a local machine.

```bash
cd code
source venv/bin/activate
python src/run.py 
# Optionally pass a different config file to the script (especially needed for the non-absa tasks)
python src/run.py --config-name other_config.yaml
# Or directly pass parameters to the script
python src/run.py dataset.name='other_dataset' experiment.run_name='other_run'
```

## Task-specific Configurations

Each task has its own configuration file in the `code/conf/` directory. For example `ssa.yaml` for Structured Sentiment Analysis.

Make sure to use the appropriate configuration file when running experiments for a specific task (see [Training](#training) for how to specify this using hydra).

## Special Tokens

Since the special tokens differ from task to task, they are stored in JSON files in the `data/special_tokens_mappings` directory. Ensure that `directories.special_tokens_mappings` in the config points to the correct directory and `dataset.special_tokens_file` points to the correct file for each task.

## Enhanced BartABSA++ Features

Our implementation includes several key improvements over the original:

### Architectural Enhancements
1. **Feature Normalization**: L2 normalization of embedding spaces + RMSNorm for training stability
2. **Parametrized Gating**: Learnable gates replacing static hyperparameters  
3. **Additional Attention**: Cross-attention mechanism reusing BART's pretrained weights

### Multi-Architecture Support
Inspired by [Rothe et al. (2020)](https://doi.org/10.1162/tacl_a_00313), we support:
- Pure encoder-decoder models (BART)
- Synthetic combinations (BERT2GPT, RoBERTa2RoBERTa, GPT22GPT2, etc.)
- Scaling experiments with various model sizes

### Reproducing Key Results

```bash
# Enhanced BartABSA++ (our main contribution)
python src/run.py model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_final_layer_norm=true

# State-of-the-art with BART-Large
python src/run.py model.base_model=facebook/bart-large model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=bart model.use_final_layer_norm=true

# Multi-architecture experiments (e.g., RoBERTa2GPT-2)
python src/run.py model.base_model=FacebookAI/roberta-base model.decoder_model=gpt2 model.gating_mode=full_gating model.normalize_encoder_outputs=true model.attention_mechanism=custom model.use_final_layer_norm=true
```

For comprehensive experiments including architecture ablations, scaling studies, and multi-task evaluation, see the batch experiment scripts in [`code/slurm/experimental_runs/`](code/slurm/experimental_runs/).

## Original BartABSA Hyperparameters

The following hyperparameters are based on the [original implementation](https://github.com/yhcc/BARTABSA/blob/main/peng/train.py):

| Hyperparameter     | Value                    |
|--------------------|--------------------------|
| Batch Size         | 16                       |
| Learning Rate      | 5e-5                     |
| Max Epochs         | 200                      |
| Early Stopping     | 30                       |
| Optimizer          | AdamW                    |
| Gradient Clip Val  | 5                        |
| Warmup Steps       | 1% of total steps        |
| LR Scheduler       | Linear Scheduler         |
| Weight Decay       | 1e-2                     |
| Sampler            | Bucket (based on source sequence length) |
| Decoding Strategy  | Beam Search (beam size 4)|

> [!WARNING]
> 1. The og implementation used beam search for decoding. Since the decoding is implemented manually in this version, currently only greedy decoding is supported.
> 2. The og implementation has a [length penalty](https://github.com/yhcc/BARTABSA/blob/13168eaaa0c22a4fe9dc18db5c743ea340edaa95/peng/train.py#L35) of 1.0. Since it's not mentioned in the original paper, it was removed.
> 3. The linear scheduler from the og implementation was replaced with the default, pytorch polynomial decay scheduler, as it seemed to perform better.
> 4. The og implementation uses a custom sampler, which was recreated using the source sequence length as the the metric.
> 5. Additionally to the `pengb` dataset the implementation also supports the [`Astev2` dataset](https://github.com/xuuuluuu/SemEval-Triplet-data), which can be specified via the `dataset.source` parameter.

These settings were used as a starting point for the experiments and may be adjusted for optimal performance in different environments.

## Cluster Running 

See [CLUSTER_RUNNING.md](CLUSTER_RUNNING.md) for more information on how to run experiments on a SLURM or Kubernetes cluster.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
tba
```

Also consider citing the original BART-ABSA work:

```bibtex
@misc{yan2021unifiedgenerativeframeworkaspectbased,
      title={A Unified Generative Framework for Aspect-Based Sentiment Analysis}, 
      author={Hang Yan and Junqi Dai and Tuo ji and Xipeng Qiu and Zheng Zhang},
      year={2021},
      eprint={2106.04300},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2106.04300}, 
}
```

## References

- [Original BARTABSA Paper and Code](https://github.com/yhcc/BARTABSA)
- [A Unified Generative Framework for Aspect-Based Sentiment Analysis (Yan et al., 2021)](https://arxiv.org/abs/2106.04300)
- [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks (Rothe et al., 2020)](https://doi.org/10.1162/tacl_a_00313)

---

## Contributing & Support

This repository is provided as-is to support reproducible research. If you find this code helpful for your research, please consider:
- ⭐ Starring this repository
- 📄 Citing our paper (see [Citation](#citation) section)
- 🐛 Opening issues for bugs or questions
- 🔧 Contributing improvements via pull requests

For questions about the original BART-ABSA method, please refer to the [original repository](https://github.com/yhcc/BARTABSA).