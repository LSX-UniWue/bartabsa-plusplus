import random
import warnings
from json import decoder
from logging import getLogger
from typing import Any, List

import numpy
import torch
from lightning import LightningDataModule
from src.dataset.absa.datasets import AsteDataset, PengDataset
from src.dataset.absa.tokenizer import ABSATokenizer
from src.dataset.datasets import AdvancedDataset, DataSplit
from src.dataset.deft.dataset import DEFTDataset
from src.dataset.deft.tokenizer import DEFTTokenizer
from src.dataset.gabsa.dataset import GABSADataset
from src.dataset.gabsa.tokenizer import GABSATokenizer
from src.dataset.gner.dataset import GNERDataset
from src.dataset.gner.tokenizer import GNERTokenizer
from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from src.dataset.spaceeval.dataset import SpaceEvalDataset
from src.dataset.spaceeval.tokenizer import SpaceEvalTokenizer
from src.dataset.sre.dataset import SREDataset
from src.dataset.sre.tokenizer import SRETokenizer
from src.dataset.ssa.dataset import SSADataset
from src.dataset.ssa.tokenizer import SSATokenizer
from src.utils.config_types import AbsaConfig
from src.utils.data_utils import BatchSamplerSimilarLength
from src.utils.model_utils import generate_tokenizer_with_special_tokens
from torch.utils.data import DataLoader
from transformers import AutoModel, EncoderDecoderModel

logger = getLogger("lightning.pytorch")


class AbsaDataModule(LightningDataModule):
    def __init__(
        self,
        config: AbsaConfig,
    ) -> None:
        assert config.directories.data, config.model.base_model
        assert config.dataset.special_tokens_config, "Special tokens mapping must be provided"
        super().__init__()

        self.absa_config = config
        self.directory = config.directories.data + f"/{config.dataset.name}" if config.dataset.name else config.directories.data
        tokenizer_data = generate_tokenizer_with_special_tokens(config.model.base_model, config.dataset.special_tokens_config.special_tokens_mapping)
        self.mapping2targetID = tokenizer_data["mapping2targetid"]
        self.mapping2ID = tokenizer_data["mapping2id"]

        if config.model.decoder_model:
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(config.model.base_model, config.model.decoder_model)
        else:
            model = AutoModel.from_pretrained(config.model.base_model)

        task = config.dataset.task
        task_config = self._get_task_config(task)

        self.tokenizer = task_config["tokenizer"](
            model,
            tokenizer_data["tokenizer"],
            tokenizer_data["mapping2id"],
            tokenizer_data["mapping2targetid"],
            config.dataset.special_tokens_config.num_labels,
            config.dataset.lang_code,
        )
        self.dataset_class = task_config["dataset"]

    def _get_task_config(self, task):
        task_configs = {
            "absa": {
                "tokenizer": ABSATokenizer,
                "dataset": PengDataset if self.absa_config.dataset.source == "pengb" else AsteDataset,
            },
            "ssa": {
                "tokenizer": SSATokenizer,
                "dataset": SSADataset,
            },
            "sre": {
                "tokenizer": SRETokenizer,
                "dataset": SREDataset,
            },
            "deft": {
                "tokenizer": DEFTTokenizer,
                "dataset": DEFTDataset,
            },
            "spaceeval": {
                "tokenizer": SpaceEvalTokenizer,
                "dataset": SpaceEvalDataset,
            },
            "gabsa": {"tokenizer": GABSATokenizer, "dataset": GABSADataset},
            "gner": {"tokenizer": GNERTokenizer, "dataset": GNERDataset},
        }

        if task not in task_configs:
            raise ValueError(f"Invalid task: {task}")

        return task_configs[task]

    def setup(self, stage: str):
        assert self.dataset_class, f"Invalid data source: {self.absa_config.dataset.source}"
        if stage == "fit":
            self.train_dataset = self.dataset_class(
                self.tokenizer, self.directory, DataSplit.TRAIN, self.absa_config.dataset.remove_duplicates, self.absa_config.dataset.fraction
            )
            self.dev_dataset = self.dataset_class(self.tokenizer, self.directory, DataSplit.DEV)
        elif stage == "test":
            self.test_dataset = self.dataset_class(self.tokenizer, self.directory, DataSplit.TEST)

    def _collate_for_decoding(self, batch):
        encoder_input_ids = [torch.tensor(item["encoder_input_ids"]) for item in batch]
        encoder_attention_mask = [torch.tensor(item["encoder_attention_mask"]) for item in batch]
        decoder_input_ids = [torch.tensor(item["decoder_input_ids"]) for item in batch]
        decoder_attention_mask = [torch.tensor(item["decoder_attention_mask"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]

        # Add missing fields
        original_sentence = [item["original_sentence"] for item in batch]
        original_labels = [item["original_labels"] for item in batch]
        sentence_raw_words = [item["sentence_raw_words"] for item in batch]
        word_mapping = [item["word_mapping"] for item in batch]

        # Pad sequences
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(encoder_input_ids, batch_first=True, padding_value=0)
        encoder_attention_mask = torch.nn.utils.rnn.pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=0)
        decoder_attention_mask = torch.nn.utils.rnn.pad_sequence(decoder_attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        max_encoder_len = encoder_input_ids.size(1)
        batch_size = encoder_input_ids.size(0)
        # TODO: Here handling for tasks with SEP and EOL tokens would be needed
        num_special_tokens = 1 + self.absa_config.dataset.special_tokens_config.num_labels  # type: ignore

        head_mask = torch.ones((batch_size, decoder_input_ids.size(1), num_special_tokens + max_encoder_len), dtype=torch.bool)
        for i, enc_len in enumerate(encoder_attention_mask.sum(1)):
            head_mask[i, :, num_special_tokens + enc_len :] = 0

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "head_mask": head_mask,
            "original_sentence": original_sentence,
            "original_labels": original_labels,
            "sentence_raw_words": sentence_raw_words,
            "word_mapping": word_mapping,
        }

    def train_dataloader(self):
        shuffle = False if self.absa_config.experiment.debug else True
        return DataLoader(
            self.train_dataset,
            batch_sampler=BatchSamplerSimilarLength(
                self.train_dataset, self.absa_config.training.batch_size, shuffle=shuffle, seed=self.absa_config.experiment.seed
            ),
            num_workers=1 if self.absa_config.experiment.debug else self.absa_config.training.num_workers,
            collate_fn=self._collate_for_decoding,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.absa_config.training.batch_size,
            num_workers=self.absa_config.training.num_workers,
            shuffle=False,
            collate_fn=self._collate_for_decoding,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.absa_config.training.batch_size,
            num_workers=self.absa_config.training.num_workers,
            shuffle=False,
            collate_fn=self._collate_for_decoding,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
        )
