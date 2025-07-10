from logging import getLogger
from typing import Optional

from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = getLogger("lightning.pytorch")


class GNERTokenizer(BaseMappingTokenizer):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mapping2ID: dict[str, int],
        mapping2targetID: dict[str, int],
        num_labels: int,
        language: str = "en_XX",
    ):
        super().__init__(model, tokenizer, mapping2ID, mapping2targetID, num_labels, language)

    def pointer_generator(
        self,
        labels: list[tuple[int, int, str]],
        offset: int,
        word_mapping: list[list[int]],
        input_encoding_ids: list[int],
        output_encoding: list[int],
    ):
        pointer_labels = []

        for _, (entity_start, entity_end, entity_type) in enumerate(labels):
            # Since the labeling is done on a word base, but we are working on a token base,
            # we always point from the first token in the word to the last token in the word.
            # This stems from the OG implementation and is not described in the paper!
            pointer_labels = pointer_labels + [
                word_mapping[entity_start][0] + offset,
                word_mapping[entity_end][-1] + offset,
                self.mapping2targetID[entity_type] + 1,
            ]
            output_encoding.extend([input_encoding_ids[i - offset] for i in pointer_labels[-3:-1]] + [self.mapping2ID[entity_type]])

        return pointer_labels, output_encoding
