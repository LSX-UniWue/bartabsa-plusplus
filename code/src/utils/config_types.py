from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import DictConfig


@dataclass
class Experiment:
    project_name: str
    entity: str
    run_name: str
    iteration: int
    seed: int
    deterministic: bool
    debug: bool
    offline: bool
    ignore_existing: bool
    write_predictions: bool
    write_heatmaps: bool
    tensors_to_plot: List[str]
    checkpoint_tmp_dir: Optional[str] = None  # Optional directory for checkpoint temp files


@dataclass
class Directories:
    data: str
    checkpoints: str
    logs: str
    predictions: str
    heatmaps: str
    special_tokens_mappings: str


@dataclass
class GradientClip:
    value: float
    algorithm: str


@dataclass
class Training:
    max_epochs: int
    batch_size: int
    num_workers: int
    precision: str
    accumulate_grad_batches: int
    check_val_every_n_epoch: int
    log_every_n_steps: int
    early_stop_patience: int
    use_last_checkpoint: bool


@dataclass
class Optimizer:
    learning_rate: float
    weight_decay: float
    gradient_clip: GradientClip


@dataclass
class Monitoring:
    metric: str
    save_top_k: int


@dataclass
class Dropout:
    general: float
    attention: float
    encoder_mlp: float


@dataclass
class Model:
    alpha: float
    base_model: str
    decoder_model: Optional[str]
    dropout: Dropout
    use_lr_scheduler: str
    use_encoder_mlp: bool
    encoder_mode: str = "default"  # Options: "default", "split_encoder", "split_encoder_simplified"
    gating_mode: str = "full_gating"  # Options: "full_gating", "encoder_gating", "decoder_gating", "no_gating"
    attention_mechanism: str = "none"  # Options: "none", "custom", "bart"
    attention_polling: str = "mean"  # Options: "mean", "max", "first_head"
    use_combined_output: bool = False
    decouple_models: bool = False  # If true the model that predicts the attention is different from the one predicting the token logits
    normalize_encoder_outputs: bool = False
    use_final_layer_norm: bool = False
    rmsnorm_eps: float = 1e-6
    use_rms_for_encoder_norm: bool = False
    dont_use_rms: bool = False
    use_dimension_normalization: bool = False
    use_value_matrix: bool = False
    randomize_encoder: bool = False
    randomize_decoder: bool = False


@dataclass
class SpecialTokensConfig:
    # Needed, since mapping2ID and mapping2targetID are not available in the config file and only added later.
    def __init__(self, num_labels: int, special_tokens_mapping: Dict[str, str], special_tokens_begl: str, special_tokens_endl: str):
        self.num_labels = num_labels
        self.special_tokens_mapping = special_tokens_mapping
        self.special_tokens_begl = special_tokens_begl
        self.special_tokens_endl = special_tokens_endl
        self.mapping2ID = None
        self.mapping2targetID = None

    num_labels: int
    special_tokens_mapping: Dict[str, str]
    special_tokens_begl: str
    special_tokens_endl: str
    mapping2ID: Optional[Dict[str, int]]
    mapping2targetID: Optional[Dict[str, int]]

    def __str__(self):
        return (
            f"SpecialTokensMapping(\n"
            f"  num_labels={self.num_labels},\n"
            f"  special_tokens_mapping={self.special_tokens_mapping},\n"
            f"  special_tokens_begl={self.special_tokens_begl},\n"
            f"  special_tokens_endl={self.special_tokens_endl}\n"
            f")"
        )


@dataclass
class Dataset:
    task: str
    source: Optional[str]
    name: Optional[str]
    remove_duplicates: bool
    fraction: float
    lang_code: str
    special_tokens_file: Optional[str]
    # The special tokens mapping is not directly given by the user, but rather imported from the file above.
    # For simplicity it is part of the Dataset (and therefore the config), to be available through the whole pipeline.
    special_tokens_config: Optional[SpecialTokensConfig]


@dataclass
class AbsaConfig:
    experiment: Experiment
    directories: Directories
    training: Training
    optimizer: Optimizer
    monitoring: Monitoring
    model: Model
    dataset: Dataset

    def __str__(self):
        return (
            f"AbsaConfig(\n"
            f"  experiment={self.experiment},\n"
            f"  directories={self.directories},\n"
            f"  training={self.training},\n"
            f"  optimizer={self.optimizer},\n"
            f"  monitoring={self.monitoring},\n"
            f"  model={self.model},\n"
            f"  dataset={self.dataset}\n"
            f")"
        )


def compare_configs(original, modified):
    """
    Compares two OmegaConf DictConfig instances and returns a dictionary of fields
    that have different values, including nested differences.
    """

    def recurse(o, m):
        diff = {}
        if isinstance(o, DictConfig) and isinstance(m, DictConfig):
            # Iterate over keys in the dictionary-like DictConfig
            for key in o:
                if key in m:
                    sub_diff = recurse(o[key], m[key])
                    if sub_diff is not None:
                        diff[key] = sub_diff
        else:
            if o != m:
                return m
        return diff if diff else None

    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict) and v:
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Calculate differences
    result = recurse(original, modified)
    # Flatten the result dictionary
    if result:
        return flatten_dict(result)
    else:
        return {}
