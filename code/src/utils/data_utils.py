import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from src.utils.config_types import SpecialTokensConfig
from torch.utils.data import Sampler


class BatchSamplerSimilarLength(Sampler):
    """
    Taken from https://discuss.pytorch.org/t/using-distributedsampler-in-combination-with-batch-sampler-to-make-sure-batches-have-sentences-of-similar-length/119824/3
    Samples batches of similar length sequences for efficient training.

    Args:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): The size of each batch.
        indices (list[int], optional): Specific indices to sample from. Defaults to None.
        shuffle (bool): Whether to shuffle the indices. Defaults to True.

    Attributes:
        batch_size (int): The size of each batch.
        shuffle (bool): Whether to shuffle the indices.
        indices (list[tuple[int, int]]): List of tuples with index and sequence length.
        pooled_indices (list[int]): List of indices sorted by sequence length.
    """

    def __init__(self, dataset, batch_size, indices=None, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(self.seed)

        # get the indices and length
        self.indices = [(i, len(sample["encoder_input_ids"])) for i, sample in enumerate(dataset)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(sorted(self.indices[i : i + self.batch_size * 100], key=lambda x: x[1]))
        self.pooled_indices = [x[0] for x in pooled_indices]

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

        # yield indices for current batch
        batches = [self.pooled_indices[i : i + self.batch_size] for i in range(0, len(self.pooled_indices), self.batch_size)]

        if self.shuffle:
            self.rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size


def import_special_tokens_mapping(task_name: str, mapping_file: Optional[str], mapping_dir: str) -> SpecialTokensConfig:
    """Import the special tokens mapping for the given dataset."""
    if mapping_file is None or mapping_file == "":
        mapping_file = f"special_tokens_{task_name}.yaml"
    with open(f"{mapping_dir}/{mapping_file}", "r") as f:
        mapping = yaml.safe_load(f)
        return SpecialTokensConfig(**mapping)


def reverse_collate_and_decode(
    batch: Dict[str, Any], predictions: torch.Tensor, num_labels: int, mapping2targetID: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Reverse the collation process to decode the model's predictions back to the original format.

    Args:
        batch (dict): The batched input data.
        predictions (torch.Tensor): The model's output predictions.

    Returns:
        list[dict]: List of dictionaries with original sentence, original labels, and decoded labels.
    """
    original_sentences = batch["original_sentence"]
    original_labels = batch["original_labels"]
    word_mappings = batch["word_mapping"]
    main_losses = batch["main_loss"]
    sentence_raw_words = batch["sentence_raw_words"]

    decoded_results = []

    for i in range(len(original_sentences)):
        padded_predictions = predictions[i]  # (num_labels)
        # Remove everything after the first 0 (EOS token)
        nonzero_indices = torch.nonzero(padded_predictions == 0, as_tuple=True)[0]
        eos_index = nonzero_indices[0].item() if len(nonzero_indices) > 0 else len(padded_predictions)
        unpadded_prediction = padded_predictions[:eos_index]

        decoded_labels = decode_predictions(unpadded_prediction.tolist(), word_mappings[i], num_labels, mapping2targetID)
        result = {
            "original_sentence": original_sentences[i],
            "original_labels": original_labels[i],
            "decoded_labels": decoded_labels,
            "had_invalids": len(unpadded_prediction.tolist()) % 5 != 0,
            "main_loss": main_losses[i].item(),
            "sentence_raw_words": sentence_raw_words[i],
        }
        decoded_results.append(result)

    return decoded_results


def decode_predictions(
    predicted_pointer_labels: List[int],
    word_mapping: List[List[int]],
    num_labels: int,
    mapping2targetid: Dict[str, int],
) -> List[Tuple[int, int, int, int, str]]:
    """
    Decodes the predicted pointer labels back to the original label format.

    Args:
        original_sentence (str): The original input sentence.
        original_labels (list[tuple[int, int, int, int, str]]): The original labeled spans.
        predicted_pointer_labels (list[int]): Predicted pointer labels from the model.
        encoder_input_ids (list[int]): Tokenized input IDs including special tokens.
        word_mapping (list[list[int]]): Mapping of word indices to token indices.

    Returns:
        list[tuple[int, int, int, int, str]]: Decoded labels in the original format.
    """
    offset = 1 + num_labels
    targetID2mapping = {v: k for k, v in mapping2targetid.items()}

    aspect_spans = []

    idx = 0
    while idx < len(predicted_pointer_labels) - 4:
        a_start_token = predicted_pointer_labels[idx] - offset
        a_end_token = predicted_pointer_labels[idx + 1] - offset
        o_start_token = predicted_pointer_labels[idx + 2] - offset
        o_end_token = predicted_pointer_labels[idx + 3] - offset
        sentiment_id = predicted_pointer_labels[idx + 4] - num_labels

        a_start_word = _find_word_index(word_mapping, a_start_token)
        a_end_word = _find_word_index(word_mapping, a_end_token)
        o_start_word = _find_word_index(word_mapping, o_start_token)
        o_end_word = _find_word_index(word_mapping, o_end_token)
        sentiment = targetID2mapping.get(sentiment_id, f"INVALID ({sentiment_id})")

        aspect_spans.append((a_start_word, a_end_word, o_start_word, o_end_word, sentiment))

        idx += 5
    return aspect_spans


def _find_word_index(word_mapping: List[List[int]], token_index: int) -> int:
    """Find the word index for a given token index."""
    return next((word_idx for word_idx, token_indices in enumerate(word_mapping) if token_index in token_indices), -1)


def generate_and_save_heatmaps(batch, logits, intermediate_tensors, sample_idx, batch_idx, mode, base_model, heatmap_dir, tensors_to_plot=None):
    model_name = base_model.split("/")[-1] if isinstance(base_model, str) else base_model[0].split("/")[-1] + "_" + base_model[1].split("/")[-1]
    path = Path(heatmap_dir) / model_name / mode
    # Make dir if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

    # Generate logits heatmap
    sample_logits = logits[sample_idx].detach().cpu().numpy()
    sample_labels = batch["labels"][sample_idx].cpu().numpy()
    head_mask = batch["head_mask"][sample_idx].detach().cpu().numpy()

    sample_logits_masked = np.ma.masked_where(np.isinf(sample_logits) | (head_mask == 0), sample_logits)

    # Filter intermediate tensors based on tensors_to_plot
    if tensors_to_plot is not None:
        intermediate_tensors = {k: v for k, v in intermediate_tensors.items() if k in tensors_to_plot}

    # Create a figure with subplots for all heatmaps
    n_intermediate = len(intermediate_tensors)
    fig, axes = plt.subplots(1, n_intermediate + 1, figsize=(5 * (n_intermediate + 1), 4))
    fig.suptitle(f"Heatmaps for {mode} (Sample {batch_idx * batch['labels'].size(0) + sample_idx}):")

    # Handle case where there's only one subplot
    if n_intermediate == 0:
        axes = [axes]

    # Plot intermediate heatmaps
    for idx, (name, tensor) in enumerate(intermediate_tensors.items()):
        im = axes[idx].imshow(tensor[sample_idx].detach().cpu().numpy(), cmap="viridis", aspect="auto")
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Dimension")
        axes[idx].set_ylabel("Sequence Position")
        plt.colorbar(im, ax=axes[idx])

    # Plot logits heatmap
    norm = plt.Normalize(vmin=np.min(sample_logits_masked), vmax=np.max(sample_logits_masked))
    im = axes[-1].imshow(sample_logits_masked.T, cmap="viridis", aspect="auto", norm=norm)
    axes[-1].set_title("Logits")
    axes[-1].set_xlabel("Sequence Position")
    axes[-1].set_ylabel("Label")
    plt.colorbar(im, ax=axes[-1], label="Normalized Logit Value")

    axes[-1].set_yticks(range(sample_logits.shape[1]))
    axes[-1].set_yticklabels(range(sample_logits.shape[1]))

    for j, label in enumerate(sample_labels):
        if label != -100:
            axes[-1].text(j, label, "X", ha="center", va="center", color="red", fontweight="bold")

    # Add a line at the end of the valid sequence
    sequence_end = np.argmax(sample_labels == -100)
    axes[-1].axvline(x=sequence_end - 0.5, color="red", linestyle="--", linewidth=2)

    plt.tight_layout()
    plt.savefig(path / f"sample_{batch_idx * batch['labels'].size(0) + sample_idx}_heatmaps.png")
    plt.close()
