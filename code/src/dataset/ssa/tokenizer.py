from logging import getLogger

from src.dataset.mapping_tokenizer import BaseMappingTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = getLogger("lightning.pytorch")


class SSATokenizer(BaseMappingTokenizer):
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
        labels: list[tuple],  # Changed to accept variable-length tuples
        offset: int,
        word_mapping: list[list[int]],
        input_encoding_ids: list[int],
        output_encoding: list[int],
    ):
        pointer_labels = []
        other_special_tokens = 1  # TODO: Move this to config, leave it as is for now

        for label in labels:
            pointer_label_span = []
            i = 0
            while i < len(label) - 1:  # -1 to exclude sentiment at the end
                if isinstance(label[i], str):  # We are at some BOS token
                    begin = label[i]
                    pointer_label_span.append(self.mapping2targetID[begin] + other_special_tokens)
                    output_encoding.append(self.mapping2ID[begin])
                    i += 1
                    while i < len(label) - 1 and isinstance(label[i], int):  # To handle split spans
                        start, end = label[i], label[i + 1]
                        pointer_label_span.extend(
                            [
                                word_mapping[start][0] + offset,
                                word_mapping[end][-1] + offset,
                            ]
                        )
                        output_encoding.extend([input_encoding_ids[j - offset] for j in pointer_label_span[-2:]])
                        i += 2
                else:
                    i += 1

            sentiment = label[-1]
            pointer_label_span.append(self.mapping2targetID[sentiment] + other_special_tokens)
            output_encoding.append(self.mapping2ID[sentiment])
            pointer_labels.extend(pointer_label_span)

        return pointer_labels, output_encoding
