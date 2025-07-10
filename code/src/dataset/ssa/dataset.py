import json
from collections import defaultdict
from functools import cmp_to_key
from logging import getLogger
from pathlib import Path

from src.dataset.datasets import AdvancedDataset, DataSplit
from src.dataset.mapping_tokenizer import BaseMappingTokenizer

logger = getLogger("lightning.pytorch")


class SSADataset(AdvancedDataset):
    def __init__(self, tokenizer: BaseMappingTokenizer, directory: str, data_split: DataSplit, *args, **kwargs):
        super().__init__(tokenizer, directory, data_split, *args, **kwargs)

    def _load_data(self):
        path = self.directory / f"{str(self.data_split)}.json"
        with open(path, "r") as f:
            return json.load(f)

    def _get_duplicate_key(self, item):
        return " ".join(item["words"])

    def _get_labels(self, item):
        return item.get("aspects", []) + item.get("opinions", []) + item.get("holders", [])

    def cmp_aspect(self, v1, v2):
        # Sort based on the first opinion part's start index
        return v1["opinion"]["from"][0] - v2["opinion"]["from"][0]

    def __getitem__(self, index):
        json_sample = self.data[index]
        words = json_sample["words"]
        aspects = json_sample["aspects"]
        opinions = json_sample["opinions"]
        holders = json_sample["holders"]

        zipped_elements = defaultdict(dict)

        for aspect in aspects:
            zipped_elements[aspect["index"]]["aspect"] = aspect
        for opinion in opinions:
            zipped_elements[opinion["index"]]["opinion"] = opinion
        for holder in holders:
            zipped_elements[holder["index"]]["holder"] = holder

        zipped_list = sorted(zipped_elements.values(), key=cmp_to_key(self.cmp_aspect))

        labels = []
        for item in zipped_list:
            label = []

            if "aspect" in item and item["aspect"]["from"][0] != -1:
                aspect = item["aspect"]
                label.append("ASP_BEGIN")
                for start, end in zip(aspect["from"], aspect["to"]):
                    label.extend([start, end - 1])

            if "opinion" in item and item["opinion"]["from"][0] != -1:
                opinion = item["opinion"]
                label.append("OPN_BEGIN")
                for start, end in zip(opinion["from"], opinion["to"]):
                    label.extend([start, end - 1])

            if "holder" in item and item["holder"]["from"][0] != -1:
                holder = item["holder"]
                label.append("HOL_BEGIN")
                for start, end in zip(holder["from"], holder["to"]):
                    label.extend([start, end - 1])

            if "aspect" in item and item["aspect"]["polarity"][0] != -1:
                label.append(aspect["polarity"])

            labels.append(tuple(label))

        sentence_string = " ".join(words)
        return self.tokenizer.tokenize_with_mapping(words, sentence_string, labels)
