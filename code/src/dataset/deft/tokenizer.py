from logging import getLogger

from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = getLogger("lightning.pytorch")


class DEFTTokenizer(BaseMappingTokenizer):
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
        labels: list[tuple[int, int, str, int, int, str, str]],
        offset: int,
        word_mapping: list[list[int]],
        input_encoding_ids: list[int],
        output_encoding: list[int],
    ):
        pointer_labels = []

        for _, (a_start, a_end, source_class, t_start, t_end, target_class, relation) in enumerate(labels):
            # Since the labeling is done on a word base, but we are working on a token base,
            # we always point from the first token in the word to the last token in the word.
            # This stems from the OG implementation and is not described in the paper!
            pointer_labels = pointer_labels + [
                word_mapping[a_start][0] + offset,
                word_mapping[a_end][-1] + offset,
                self.mapping2targetID[source_class] + 1,
                word_mapping[t_start][0] + offset,
                word_mapping[t_end][-1] + offset,
                self.mapping2targetID[target_class] + 1,
                self.mapping2targetID[relation] + 1,
            ]
            output_encoding.extend(
                [input_encoding_ids[i - offset] for i in pointer_labels[-7:-5]]
                + [self.mapping2ID[source_class]]
                + [input_encoding_ids[i - offset] for i in pointer_labels[-4:-2]]
                + [self.mapping2ID[target_class]]
                + [self.mapping2ID[relation]]
            )

        return pointer_labels, output_encoding
