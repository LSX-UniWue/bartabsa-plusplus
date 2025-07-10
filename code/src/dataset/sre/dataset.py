import json
from collections import defaultdict
from functools import cmp_to_key
from logging import getLogger
from pathlib import Path

from src.dataset.datasets import AdvancedDataset, DataSplit, WueNLPDataset
from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from src.utils.task_utils import cmp_aspect
from wuenlp.impl.UIMANLPStructs import AnnotationList, UIMADocument, UIMASentence, UIMASentimentTuple

logger = getLogger("lightning.pytorch")


class SREDataset(WueNLPDataset):
    def __init__(self, tokenizer: BaseMappingTokenizer, directory: str, data_split: DataSplit, *args, **kwargs):
        super().__init__(tokenizer, directory, data_split, *args, **kwargs)
        self._remove_overlong_samples()

    def __getitem__(self, index):
        """
        This is basically the same as the ABSA XMI dataset handling.
        """
        sentence: UIMASentence = self.data[index]
        sentence_string = " ".join(list(map(lambda t: t.text, sentence.tokens)))
        sentence_raw_words = [token.text for token in sentence.tokens]
        sentiment_tuples = sentence.covered(UIMASentimentTuple)

        labels = []
        for sentiment_tuple in sentiment_tuples:
            source_s = sentiment_tuple.source.token_begin_within(sentence)
            source_e = sentiment_tuple.source.token_end_within(sentence)
            target_s = sentiment_tuple.target.token_begin_within(sentence)
            target_e = sentiment_tuple.target.token_end_within(sentence)
            label = (source_s, source_e, target_s, target_e, sentiment_tuple.sentiment)
            labels.append(label)

        return self.tokenizer.tokenize_with_mapping(sentence_raw_words, sentence_string, labels)
