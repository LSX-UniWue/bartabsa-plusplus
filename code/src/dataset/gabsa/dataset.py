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


class GABSADataset(WueNLPDataset):
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
        entities = sorted(entities, key=cmp_to_key(cmp_aspect_entity))  # type: ignore

        labels = []
        for entity in entities:
            if not (entity.begin == entity.end and entity.begin == sentence.begin):
                aspect_s = entity.token_begin_within(sentence)
                aspect_e = entity.token_end_within(sentence)
            else:
                aspect_s = None
                aspect_e = None
            types = entity.reference_type.split("#")
            aspect_class = types[0]
            aspect_sentiment = types[1]
            label = (aspect_s, aspect_e, aspect_class, aspect_sentiment)
            labels.append(label)

        return self.tokenizer.tokenize_with_mapping(sentence_raw_words, sentence_string, labels)
