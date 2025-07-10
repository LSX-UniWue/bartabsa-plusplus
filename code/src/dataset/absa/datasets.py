import json
from collections import Counter, defaultdict
from enum import Enum
from functools import cmp_to_key
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from src.dataset.datasets import AdvancedDataset, DataSplit, WueNLPDataset
from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from src.utils.task_utils import cmp_aspect
from torch.utils.data import Dataset
from wuenlp.impl.UIMANLPStructs import AnnotationList, UIMADocument, UIMASentence, UIMASentimentTuple

logger = getLogger("lightning.pytorch")


class PengDataset(AdvancedDataset):
    def __init__(self, tokenizer: BaseMappingTokenizer, directory: str, data_split: DataSplit, *args, **kwargs):
        self.directory = Path(directory)
        self.data_split = data_split
        self.use_json = self._detect_file_type()
        logger.info(f"Detected file type: {'JSON' if self.use_json else 'XMI'}")
        if not self.use_json:
            self.wuenlp_dataset = WueNLPDataset(tokenizer, directory, data_split, *args, **kwargs)
        super().__init__(tokenizer, directory, data_split, *args, **kwargs)

    def _detect_file_type(self):
        json_path = self.directory / f"{str(self.data_split)}.json"
        xmi_path = self.directory / f"{str(self.data_split)}.xmi"

        if json_path.exists():
            return True
        elif xmi_path.exists():
            return False
        else:
            raise FileNotFoundError(f"No valid data file found for {self.data_split} in {self.directory}")

    def _load_data(self):
        if self.use_json:
            return self._load_json_data()
        else:
            return self.wuenlp_dataset._load_data()

    def _load_json_data(self):
        path = self.directory / f"{str(self.data_split)}.json"
        with open(path, "r") as f:
            return json.load(f)

    def _get_duplicate_key(self, item):
        if self.use_json:
            return " ".join(item["words"])
        else:
            return self.wuenlp_dataset._get_duplicate_key(item)

    def _get_labels(self, item):
        if self.use_json:
            return item["aspects"]
        else:
            return self.wuenlp_dataset._get_labels(item)

    def __getitem__(self, index):
        if self.use_json:
            return self._get_json_item(index)
        else:
            return self._get_xmi_item(index)

    def _get_json_item(self, index):
        json_sample = self.data[index]
        words = json_sample["words"]
        aspects = json_sample["aspects"]
        opinions = json_sample["opinions"]
        assert len(aspects) == len(opinions)

        aspect_opinion_pairs = list(zip(aspects, opinions))
        aspect_opinion_pairs.sort(key=lambda x: x[0]["from"])

        labels = []
        for aspect, opinion in aspect_opinion_pairs:
            aspect_s, aspect_e = aspect["from"], aspect["to"] - 1
            opinion_s, opinion_e = opinion["from"], opinion["to"] - 1
            sentiment = aspect["polarity"]
            labels.append((aspect_s, aspect_e, opinion_s, opinion_e, sentiment))

        sentence_string = " ".join(words)
        return self.tokenizer.tokenize_with_mapping(words, sentence_string, labels)

    def _get_xmi_item(self, index):
        sentence: UIMASentence = self.data[index]
        sentence_string = " ".join(list(map(lambda t: t.text, sentence.tokens)))
        sentence_raw_words = [token.text for token in sentence.tokens]
        sentiment_tuples = sentence.covered(UIMASentimentTuple)
        sentiment_tuples = sorted(sentiment_tuples, key=cmp_to_key(cmp_aspect))

        labels = []
        for sentiment_tuple in sentiment_tuples:
            aspect_s = sentiment_tuple.expression.token_begin_within(sentence)
            aspect_e = sentiment_tuple.expression.token_end_within(sentence)
            opinion_s = sentiment_tuple.target.token_begin_within(sentence)
            opinion_e = sentiment_tuple.target.token_end_within(sentence)
            label = (aspect_s, aspect_e, opinion_s, opinion_e, sentiment_tuple.sentiment)
            labels.append(label)

        return self.tokenizer.tokenize_with_mapping(sentence_raw_words, sentence_string, labels)


class AsteDataset(AdvancedDataset):
    def _load_data(self):
        path = self.directory / f"{str(self.data_split)}.txt"
        with open(path, "r") as f:
            return [line.strip().split("####") for line in f]

    def _get_duplicate_key(self, item):
        return item[0]  # The sentence is the first element in the item

    def _get_labels(self, item):
        return eval(item[1])  # The labels are the second element in the item

    def __getitem__(self, index):
        sentence, labels_str = self.data[index]
        words = sentence.split()
        labels = eval(labels_str)

        processed_labels = []
        for aspect_pos, opinion_pos, sentiment in labels:
            aspect_s, aspect_e = aspect_pos[0], aspect_pos[-1]
            opinion_s, opinion_e = opinion_pos[0], opinion_pos[-1]
            processed_labels.append((aspect_s, aspect_e, opinion_s, opinion_e, sentiment))

        return self.tokenizer.tokenize_with_mapping(words, sentence, processed_labels)
