import json
from collections import defaultdict
from functools import cmp_to_key
from logging import getLogger
from pathlib import Path

from src.dataset.datasets import AdvancedDataset, DataSplit, WueNLPDataset
from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from src.utils.task_utils import cmp_aspect, cmp_aspect_entity
from wuenlp.impl.UIMANLPStructs import AnnotationList, UIMADocument, UIMAEntityReference, UIMASentence, UIMASentimentTuple

logger = getLogger("lightning.pytorch")


class GNERDataset(WueNLPDataset):
    def __init__(self, tokenizer: BaseMappingTokenizer, directory: str, data_split: DataSplit, *args, **kwargs):
        super().__init__(tokenizer, directory, data_split, *args, **kwargs)
        self._remove_overlong_samples(max_length=512)
        self.get_dataset_stats()

    def __getitem__(self, index):
        """
        This is basically the same as the ABSA XMI dataset handling.
        """
        sentence: UIMASentence = self.data[index]
        sentence_string = " ".join(list(map(lambda t: t.text, sentence.tokens)))
        sentence_raw_words = [token.text for token in sentence.tokens]
        entities = sentence.covered(UIMAEntityReference)
        # entities = {f"b{e.begin}e{e.end}t{e.reference_type}": e for e in entities}.values()

        labels = []
        for entity in entities:
            entity_s = entity.token_begin_within(sentence)
            entity_e = entity.token_end_within(sentence)
            label = (entity_s, entity_e, entity.reference_type)
            labels.append(label)

        return self.tokenizer.tokenize_with_mapping(sentence_raw_words, sentence_string, labels)
