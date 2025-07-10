from logging import getLogger

from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = getLogger("lightning.pytorch")


class SpaceEvalTokenizer(BaseMappingTokenizer):
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

        for _, (e_start, e_end, ref_type) in enumerate(labels):
            # Since the labeling is done on a word base, but we are working on a token base,
            # we always point from the first token in the word to the last token in the word.
            # This stems from the OG implementation and is not described in the paper!
            pointer_labels = pointer_labels + [
                word_mapping[e_start][0] + offset,
                word_mapping[e_end][-1] + offset,
                self.mapping2targetID[ref_type] + 1,
            ]
            output_encoding.extend([input_encoding_ids[i - offset] for i in pointer_labels[-3:-1]] + [self.mapping2ID[ref_type]])

        return pointer_labels, output_encoding
