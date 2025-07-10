from logging import getLogger
from typing import Optional

from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = getLogger("lightning.pytorch")


class GABSATokenizer(BaseMappingTokenizer):
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
        labels: list[tuple[Optional[int], Optional[int], str, str]],
        offset: int,
        word_mapping: list[list[int]],
        input_encoding_ids: list[int],
        output_encoding: list[int],
    ):
        pointer_labels = []

        for _, (a_start, a_end, aspect_class, aspect_sentiment) in enumerate(labels):
            # Since the labeling is done on a word base, but we are working on a token base,
            # we always point from the first token in the word to the last token in the word.
            # This stems from the OG implementation and is not described in the paper!
            if a_start is not None and a_end is not None:
                pointer_labels = pointer_labels + [
                    word_mapping[a_start][0] + offset,
                    word_mapping[a_end][-1] + offset,
                    self.mapping2targetID[aspect_class] + 1,
                    self.mapping2targetID[aspect_sentiment] + 1,
                    self.mapping2targetID["SEP"] + 1,
                ]
                output_encoding.extend(
                    [input_encoding_ids[i - offset] for i in pointer_labels[-5:-3]]
                    + [self.mapping2ID[aspect_class], self.mapping2ID[aspect_sentiment], self.mapping2ID["SEP"]]
                )
            else:
                pointer_labels = pointer_labels + [
                    self.mapping2targetID[aspect_class] + 1,
                    self.mapping2targetID[aspect_sentiment] + 1,
                    self.mapping2targetID["SEP"] + 1,
                ]
                output_encoding.extend([self.mapping2ID[aspect_class], self.mapping2ID[aspect_sentiment], self.mapping2ID["SEP"]])

        return pointer_labels, output_encoding
