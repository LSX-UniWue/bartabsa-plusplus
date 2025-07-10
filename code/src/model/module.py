import json
import math
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from lightning import LightningModule
from src.metrics.absa_metric import ABSAMetric
from src.metrics.deft_metric import DEFTMetric
from src.metrics.gabsa_metric import GabsaMetric
from src.metrics.gner_metric import GnerMetric
from src.metrics.spaceeval_metric import SpaceEvalMetric
from src.metrics.sre_metric import SREMetric
from src.metrics.ssa_metric import SSAMetric
from src.metrics.triplet_metric import TripletMetric
from src.utils.config_types import AbsaConfig
from src.utils.data_utils import generate_and_save_heatmaps, reverse_collate_and_decode
from src.utils.wue_nlp_utils import convert_to_xmi
from torch.nn import functional as F
from torch.optim import AdamW  # type: ignore
from torch.optim.lr_scheduler import PolynomialLR

from .model import AbsaEncoderDecoderModel

logger = getLogger("lightning.pytorch")

# Constants not worth moving to config
PREDICTION_FILENAME = "predictions.json"
PREDICTION_XMI_FILENAME = "predictions.xmi"


class AbsaEncoderDecoderModule(LightningModule):
    def __init__(self, config: AbsaConfig, logs_dir: str):
        super().__init__()
        self.save_hyperparameters()
        self.absa_config = config

        self.model = AbsaEncoderDecoderModel(config)
        self._setup_metric()

        if config.experiment.write_predictions and self.absa_config.dataset.task != "absa":
            logger.warning("Saving predictions is currently only supported for ABSA. Setting write_predictions to False.")
            self.save_predictions = False

        self.save_predictions = config.experiment.write_predictions
        if self.save_predictions:
            self.predicted_batches_buffer: List[Dict[str, Any]] = []
            self.predictions_path = Path(logs_dir) / config.directories.predictions

    def setup(self, stage: Optional[str] = None) -> None:
        self.metric.reset()

        # Initialize gate statistics collectors for epoch-level analysis
        # Using lists of flattened values rather than tensors of different shapes
        self.encoder_gate_values_flat = []
        self.attention_gate_values_flat = []

    def _setup_metric(self):
        assert self.absa_config.dataset.special_tokens_config is not None, "Special tokens config is required, but not provided."
        if self.absa_config.dataset.task == "absa":
            # self.metric = ABSAMetric(self.absa_config.dataset.special_tokens_config.mapping2targetID, 0)
            self.metric = TripletMetric(0, self.absa_config.dataset.special_tokens_config.num_labels)
        elif self.absa_config.dataset.task == "ssa":
            self.metric = SSAMetric(self.absa_config.dataset.special_tokens_config.mapping2targetID, 0)
        elif self.absa_config.dataset.task == "sre":
            self.metric = SREMetric(self.absa_config.dataset.special_tokens_config.mapping2targetID, 0)
        elif self.absa_config.dataset.task == "deft":
            self.metric = DEFTMetric(self.absa_config.dataset.special_tokens_config.mapping2targetID, 0)
        elif self.absa_config.dataset.task == "spaceeval":
            self.metric = SpaceEvalMetric(self.absa_config.dataset.special_tokens_config.mapping2targetID, 0)
        elif self.absa_config.dataset.task == "gabsa":
            self.metric = GabsaMetric(self.absa_config.dataset.special_tokens_config.mapping2targetID, 0)
        elif self.absa_config.dataset.task == "gner":
            self.metric = GnerMetric(self.absa_config.dataset.special_tokens_config.mapping2targetID, 0)
        else:
            raise ValueError(f"Unknown task: {self.absa_config.dataset.task}")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits, loss = self._forward_common_step(batch, batch_idx, "train")
        self._evaluate_and_log(logits.argmax(dim=-1), batch["labels"], "train")
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self._generate_common_step(batch, batch_idx, "dev")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self._generate_common_step(batch, batch_idx, "test")

    def _forward_common_step(self, batch: Dict[str, Any], batch_idx: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = batch["labels"].size(0)
        logits, intermediate_tensors = self.model.forward(batch)

        # Enhanced gating mechanism logging
        if "encoder_gating_gate" in intermediate_tensors:
            encoder_gate = intermediate_tensors["encoder_gating_gate"]

            # Store flattened values for epoch-level analysis if in dev or test mode
            if mode == "dev" or mode == "test":
                # Apply mask to only include real tokens (not padding)
                # Use encoder attention mask to filter out padding tokens
                mask = batch["encoder_attention_mask"].bool()
                # Only keep values where mask is True and then flatten
                masked_values = encoder_gate[mask]
                self.encoder_gate_values_flat.append(masked_values.detach().cpu().float())

            # Only log the most important statistics (mean and std)
            # Apply mask when calculating mean and std
            mask = batch["encoder_attention_mask"].bool()
            masked_encoder_gate = encoder_gate[mask]
            self.log(f"{mode}/encoder_gate_mean", masked_encoder_gate.mean(), on_step=False, on_epoch=True, batch_size=bsz)
            self.log(f"{mode}/encoder_gate_std", masked_encoder_gate.std(), on_step=False, on_epoch=True, batch_size=bsz)

            # Log histogram only occasionally to avoid cluttering
            flat_data = masked_encoder_gate.detach().cpu().float().numpy().tolist()
            histogram_data = wandb.Histogram(flat_data)
            self._log_to_wandb({f"{mode}/encoder_gate_histogram": histogram_data})

        if "attention_gating_gate" in intermediate_tensors:
            attention_gate = intermediate_tensors["attention_gating_gate"]

            # Store flattened values for epoch-level analysis if in dev or test mode
            if mode == "dev" or mode == "test":
                # Apply mask to only include real tokens (not padding)
                # Use decoder attention mask to filter out padding tokens
                mask = batch["decoder_attention_mask"].bool()
                # Only keep values where mask is True and then flatten
                masked_values = attention_gate[mask]
                self.attention_gate_values_flat.append(masked_values.detach().cpu().float())

            # Only log the most important statistics (mean and std)
            # Apply mask when calculating mean and std
            mask = batch["decoder_attention_mask"].bool()
            masked_attention_gate = attention_gate[mask]
            self.log(f"{mode}/attention_gate_mean", masked_attention_gate.mean(), on_step=False, on_epoch=True, batch_size=bsz)
            self.log(f"{mode}/attention_gate_std", masked_attention_gate.std(), on_step=False, on_epoch=True, batch_size=bsz)

            flat_data = masked_attention_gate.detach().cpu().float().numpy().tolist()
            histogram_data = wandb.Histogram(flat_data)
            self._log_to_wandb({f"{mode}/attention_gate_histogram": histogram_data})

        if self.absa_config.experiment.write_heatmaps:
            for sample_idx in range(bsz):
                generate_and_save_heatmaps(
                    batch,
                    logits,
                    intermediate_tensors,
                    sample_idx,
                    batch_idx,
                    mode,
                    self.absa_config.model.base_model,
                    self.absa_config.directories.heatmaps,
                    self.absa_config.experiment.tensors_to_plot,
                )
        main_loss = F.cross_entropy(logits.transpose(1, 2), batch["labels"], reduction="none")
        main_loss = main_loss[batch["labels"] != -100].mean()
        self.log(f"{mode}/loss", main_loss, on_step=True, on_epoch=True, batch_size=bsz)
        return logits, main_loss

    def _generate_common_step(self, batch: Dict[str, Any], batch_idx: int, mode: str) -> None:
        """Common step for validation and test steps."""
        self._forward_common_step(batch, batch_idx, mode)

        longest_sequence = batch["decoder_input_ids"].size(0) * 2

        batch["decoder_input_ids"] = batch["decoder_input_ids"][:, 0].unsqueeze(-1)
        batch["decoder_attention_mask"] = batch["decoder_attention_mask"][:, 0].unsqueeze(-1)

        prediction = self.model.generate(batch, longest_sequence)
        target = batch["labels"]
        self._evaluate_and_log(prediction, target, mode)

        if mode == "test" and self.save_predictions:
            self.predicted_batches_buffer.append({"batch": batch, "prediction": prediction})

    def _evaluate_and_log(self, prediction: torch.Tensor, target: torch.Tensor, mode: str) -> None:
        self.metric.evaluate(prediction, target)
        metrics = self.metric.get_metrics()
        self.log_dict({f"{mode}/{k}": v for k, v in metrics.items()}, on_step=True, on_epoch=True, batch_size=target.size(0))

    def _decode_and_save_predictions(self) -> None:
        assert self.absa_config.dataset.special_tokens_config is not None, "Special tokens config is required for decoding predictions"
        assert self.absa_config.dataset.special_tokens_config.mapping2targetID is not None, "Mapping2targetID is required for decoding predictions"
        assert self.absa_config.dataset.special_tokens_config.mapping2ID is not None, "Mapping2ID is required for decoding predictions"
        decoded_predictions = []
        logger.info(f"Decoding predictions for {len(self.predicted_batches_buffer)} batches.")
        for item in self.predicted_batches_buffer:
            batch, predictions = item["batch"], item["prediction"]
            decoded_results = reverse_collate_and_decode(
                batch,
                predictions,
                self.absa_config.dataset.special_tokens_config.num_labels,
                self.absa_config.dataset.special_tokens_config.mapping2targetID,
            )
            for decoded_result in decoded_results:
                single_prediction = {
                    "raw_words": decoded_result["original_sentence"],
                    "words": decoded_result["sentence_raw_words"],
                    "original_labels": decoded_result["original_labels"],
                    "decoded_labels": decoded_result["decoded_labels"],
                    "had_invalids": decoded_result["had_invalids"],
                    "main_loss": decoded_result["main_loss"],
                }
                decoded_predictions.append(single_prediction)

        logger.info(f"Writing predictions to {self.predictions_path}")
        self.predictions_path.mkdir(parents=True, exist_ok=True)
        with open(self.predictions_path / PREDICTION_FILENAME, "w") as f:
            json.dump(decoded_predictions, f)
        convert_to_xmi(decoded_predictions, self.predictions_path / PREDICTION_XMI_FILENAME)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.absa_config.optimizer.learning_rate,
            weight_decay=self.absa_config.optimizer.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler_config = {"scheduler": PolynomialLR(optimizer, total_iters=total_steps, power=2.0), "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()
        # Reset gate value collectors
        self.encoder_gate_values_flat = []
        self.attention_gate_values_flat = []

    def on_validation_epoch_end(self) -> None:
        self.metric.reset()
        # Analyze collected gate statistics if available
        self._analyze_and_log_gate_statistics("dev")

    def on_test_epoch_start(self) -> None:
        self.metric.reset()
        # Reset gate value collectors
        self.encoder_gate_values_flat = []
        self.attention_gate_values_flat = []

    def on_test_epoch_end(self) -> None:
        self.metric.reset()
        # Analyze collected gate statistics if available
        self._analyze_and_log_gate_statistics("test")

    def on_test_end(self) -> None:
        self.metric.reset()
        if self.save_predictions:
            logger.info("Testing finished, decoding and saving predictions")
            self._decode_and_save_predictions()
        else:
            logger.info("Testing finished, no predictions saved")

    def _log_to_wandb(self, data: Dict[str, Any]) -> None:
        """Safely log data to wandb if available"""
        try:
            if self.logger is not None:
                # Access experiment attribute safely
                experiment = getattr(self.logger, "experiment", None)
                if experiment is not None and hasattr(experiment, "log"):
                    experiment.log(data)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def _create_gate_heatmap(self, gate_values: torch.Tensor, title: str, batch_idx: int) -> Optional[wandb.Image]:
        """Create a heatmap visualization for gate values if dependencies are available"""
        if self.logger is None:
            return None

        try:
            # Get the first sample in the batch
            sample_gate = gate_values[0].detach().cpu().numpy()

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 3))
            im = ax.imshow(sample_gate, cmap="viridis", aspect="auto")
            ax.set_title(f"{title} (Sample 0, Batch {batch_idx})")
            ax.set_ylabel("Sequence Position")
            ax.set_xlabel("Hidden Dimension")
            fig.colorbar(im)

            image = wandb.Image(fig)
            plt.close(fig)
            return image
        except Exception as e:
            logger.warning(f"Failed to create gate heatmap: {e}")
            return None

    def _create_token_level_heatmap(self, gate_values: torch.Tensor, input_ids: torch.Tensor, title: str, batch_idx: int) -> Optional[wandb.Image]:
        """Create a token-level heatmap showing gate values per token"""
        if self.logger is None:
            return None

        try:
            # Get the first sample in the batch for visualization
            sample_gate = gate_values[0].detach().cpu().numpy()  # shape: [seq_len, hidden_dim]
            sample_input_ids = input_ids[0].detach().cpu().numpy()

            # Calculate mean gate value per token
            token_gate_means = sample_gate.mean(axis=1)  # Average across hidden dimension

            # Get token text to display (limited to first 30 tokens for visibility)
            max_tokens = min(30, len(sample_input_ids))
            token_texts = []
            for i in range(max_tokens):
                try:
                    token_texts.append(self.model.tokenizer.decode([sample_input_ids[i]]))
                except:
                    token_texts.append("[UNK]")

            # Truncate for visualization
            token_gate_means = token_gate_means[:max_tokens]

            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, max(5, max_tokens // 4)))
            bars = ax.barh(range(len(token_texts)), token_gate_means, color="skyblue")
            ax.set_yticks(range(len(token_texts)))
            ax.set_yticklabels(token_texts)
            ax.set_title(f"{title} Per Token (Sample 0, Batch {batch_idx})")
            ax.set_xlabel("Mean Gate Value")
            ax.set_ylabel("Token")

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width + 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center")

            plt.tight_layout()
            image = wandb.Image(fig)
            plt.close(fig)
            return image
        except Exception as e:
            logger.warning(f"Failed to create token-level gate heatmap: {e}")
            return None

    def _analyze_and_log_gate_statistics(self, mode: str) -> None:
        """Analyze gate statistics across the entire epoch and log summary visualizations"""
        # Process encoder gate statistics
        if len(self.encoder_gate_values_flat) > 0:
            try:
                # Concatenate all collected values
                all_encoder_gates = torch.cat(self.encoder_gate_values_flat, dim=0)

                # Overall histogram - this is the most important visualization
                flat_data = all_encoder_gates.detach().cpu().float().numpy().tolist()
                histogram = wandb.Histogram(flat_data)
                self._log_to_wandb({f"{mode}/epoch_encoder_gate_histogram": histogram})

                # Log basic summary statistics
                self._log_to_wandb(
                    {
                        f"{mode}/epoch_encoder_gate_mean": float(all_encoder_gates.mean()),
                        f"{mode}/epoch_encoder_gate_std": float(all_encoder_gates.std()),
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to analyze encoder gate statistics: {e}")

        # Process attention gate statistics
        if len(self.attention_gate_values_flat) > 0:
            try:
                # Concatenate all collected values
                all_attention_gates = torch.cat(self.attention_gate_values_flat, dim=0)

                # Overall histogram - this is the most important visualization
                flat_data = all_attention_gates.detach().cpu().float().numpy().tolist()
                histogram = wandb.Histogram(flat_data)
                self._log_to_wandb({f"{mode}/epoch_attention_gate_histogram": histogram})

                # Log basic summary statistics
                self._log_to_wandb(
                    {
                        f"{mode}/epoch_attention_gate_mean": float(all_attention_gates.mean()),
                        f"{mode}/epoch_attention_gate_std": float(all_attention_gates.std()),
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to analyze attention gate statistics: {e}")
