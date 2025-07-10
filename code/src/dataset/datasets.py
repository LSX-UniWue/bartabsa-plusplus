import json
from abc import abstractmethod
from collections import Counter, defaultdict
from enum import Enum
from functools import cmp_to_key
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from src.utils.task_utils import cmp_aspect
from torch.utils.data import Dataset
from wuenlp.impl.UIMANLPStructs import AnnotationList, UIMADocument, UIMASentence, UIMASentimentTuple

logger = getLogger("lightning.pytorch")


class DataSplit(str, Enum):
    TRAIN = "train"
    DEV = "valid"
    TEST = "test"

    def __str__(self):
        return self.value


class AdvancedDataset(Dataset):
    """
    Dataset class usable for all tasks. Implements duplicate removal and dataset reduction.

    Subclass must implement:
    - _load_data()
    - _get_duplicate_key()
    - _get_labels()
    - __getitem__()
    """

    def __init__(
        self, tokenizer: BaseMappingTokenizer, directory: str, data_split: DataSplit, remove_duplicates: bool = True, dataset_fraction: float = 1.0
    ):
        assert tokenizer is not None, "Tokenizer must not be None"
        assert directory is not None, "Directory must not be None"
        assert data_split is not None, "DataSplit must not be None"
        assert 0.0 < dataset_fraction <= 1.0, "Dataset fraction must be greater than 0 and less or equal to 1"

        self.tokenizer = tokenizer
        self.directory = Path(directory)
        self.data_split = data_split
        self.remove_duplicates = remove_duplicates
        self.dataset_fraction = dataset_fraction
        self.match_statistics = Counter()

        self.data = self._load_data()

        if self.data_split == DataSplit.TRAIN and self.remove_duplicates:
            self._remove_duplicates()

        if self.data_split == DataSplit.TRAIN and self.dataset_fraction < 1.0:
            self._reduce_dataset()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError("This method must be implemented by a subclass")

    def _remove_duplicates(self):
        if self.data_split != DataSplit.TRAIN:
            return

        sentence_dict = defaultdict(list)
        for i, item in enumerate(self.data):
            key = self._get_duplicate_key(item)
            sentence_dict[key].append((i, item))

        filtered_data = []
        for _, value in sentence_dict.items():
            if len(value) > 1:
                # Keep the item with the most labels to avoid loss of information
                max_labels = max(value, key=lambda x: len(self._get_labels(x[1])))
                filtered_data.append(max_labels[1])
            else:
                filtered_data.append(value[0][1])

        logger.info(f"Filtered out {len(self.data) - len(filtered_data)} duplicate sentences")
        self.data = filtered_data

    @abstractmethod
    def _get_duplicate_key(self, item):
        raise NotImplementedError("This method must be implemented by a subclass")

    @abstractmethod
    def _get_labels(self, item):
        """
        Only used for duplicate removal. (Keeps the item with the most labels)
        """
        raise NotImplementedError("This method must be implemented by a subclass")

    def _reduce_dataset(self):
        original_size = len(self.data)
        logger.info(f"Reducing dataset size from {original_size} to {int(original_size * self.dataset_fraction)}")
        generator = torch.Generator()
        indices = torch.randperm(original_size, generator=generator)[: int(original_size * self.dataset_fraction)]
        self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("This method must be implemented by a subclass")


class WueNLPDataset(AdvancedDataset):
    """
    Dataset for tasks in WueNLP XMI format.

    Every task has to implement the __getitem__() method.
    """

    def __init__(self, tokenizer: BaseMappingTokenizer, directory: str, data_split: DataSplit, *args, **kwargs):
        super().__init__(tokenizer, directory, data_split, *args, **kwargs)

    def _load_data(self):
        path = self.directory / f"{str(self.data_split)}.xmi"
        document = UIMADocument.from_xmi(path)
        return document.sentences

    def _remove_overlong_samples(self, max_length: int = 512):
        """
        Remove samples that are too long for the model, i.e. aren't solvable with this design anyways.
        """
        sentences_to_remove = []
        for sample in self.data:
            sentence: UIMASentence = sample
            tokens = []
            for token in sentence.tokens:
                tokens.extend(self.tokenizer.tokenizer.tokenize(token.text))
            if len(tokens) + 2 > max_length:
                sentences_to_remove.append(sentence)

        logger.info(f"Removed {len(sentences_to_remove)} overlong sentences from {self.data_split} split")
        self.data = [sample for sample in self.data if sample not in sentences_to_remove]

    def get_dataset_stats(self):
        num_sentences = len(self.data)
        num_tokens = sum(len(sentence.tokens) for sentence in self.data)
        avg_tokens_per_sentence = num_tokens / num_sentences if num_sentences > 0 else 0
        longest_sequence_length = max(len(sentence.tokens) for sentence in self.data)
        num_long_sentences = sum(1 for sentence in self.data if len(sentence.tokens) > 512)

        logger.info(f"Stats for {self.data_split} split")
        logger.info(f"Number of sentences: {num_sentences}")
        logger.info(f"Total number of tokens: {num_tokens}")
        logger.info(f"Average tokens per sentence: {avg_tokens_per_sentence:.2f}")
        logger.info(f"Longest sequence length: {longest_sequence_length}")
        logger.info(f"Number of sentences over 512 tokens: {num_long_sentences}")

        return {
            "num_sentences": num_sentences,
            "num_tokens": num_tokens,
            "avg_tokens_per_sentence": avg_tokens_per_sentence,
            "longest_sequence_length": longest_sequence_length,
        }

    def _get_duplicate_key(self, item):
        return item.text

    def _get_labels(self, item):
        return item.covered(UIMASentimentTuple)
