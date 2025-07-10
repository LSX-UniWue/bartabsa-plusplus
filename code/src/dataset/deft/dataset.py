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


class DEFTDataset(WueNLPDataset):
    def __init__(self, tokenizer: BaseMappingTokenizer, directory: str, data_split: DataSplit, *args, **kwargs):
        super().__init__(tokenizer, directory, data_split, *args, **kwargs)
        self._remove_overlong_samples()
        self.get_dataset_stats()

    def _load_data(self):
        """For DEFT we have the special case that we have multiple XMI files for each split."""
        path = self.directory / f"{str(self.data_split)}"
        files = list(path.glob("*.xmi"))
        all_sentences: list[UIMASentence] = []
        for file in files:
            document = UIMADocument.from_xmi(file)
            all_sentences.extend(document.sentences)
        return all_sentences

    def __getitem__(self, index):
        """
        This is basically the same as the ABSA XMI dataset handling.
        """
        sentence: UIMASentence = self.data[index]
        sentence_string = " ".join(list(map(lambda t: t.text, sentence.tokens)))
        sentence_raw_words = [token.text for token in sentence.tokens]
        sentiment_tuples = sentence.covered(UIMASentimentTuple)
        sentiment_tuples = sorted(sentiment_tuples, key=cmp_to_key(cmp_aspect))

        labels = []
        for sentiment_tuple in sentiment_tuples:
            source_s = sentiment_tuple.source.token_begin_within(sentence)
            source_e = sentiment_tuple.source.token_end_within(sentence)
            source_class = sentiment_tuple.source.reference_type
            target_s = sentiment_tuple.target.token_begin_within(sentence)
            target_e = sentiment_tuple.target.token_end_within(sentence)
            target_class = sentiment_tuple.target.reference_type
            label = (source_s, source_e, source_class, target_s, target_e, target_class, sentiment_tuple.sentiment)
            labels.append(label)

        return self.tokenizer.tokenize_with_mapping(sentence_raw_words, sentence_string, labels)
