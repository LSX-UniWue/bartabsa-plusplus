from logging import getLogger

from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore

logger = getLogger("lightning.pytorch")


class BaseMappingTokenizer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mapping2ID: dict[str, int],
        mapping2targetID: dict[str, int],
        num_labels: int,
        language: str = "en_XX",
    ):
        assert tokenizer is not None, "Tokenizer must not be None"
        assert mapping2ID is not None, "Mapping2ID must not be None"
        assert mapping2targetID is not None, "Mapping2TargetID must not be None"

        self.model = model
        self.tokenizer = tokenizer
        self.mapping2targetID = mapping2targetID
        self.mapping2ID = mapping2ID
        self.num_labels = num_labels
        self.language = language

    def pointer_generator(
        self,
        labels: list[tuple],
        offset: int,
        word_mapping: list[list[int]],
        input_encoding_ids: list[int | None],
        output_encoding: list[int],
    ):
        raise NotImplementedError("This method must be implemented by a subclass")

    def tokenize_with_mapping(
        self, sentence_raw_words: list[str], sentence: str, labels: list[tuple[int, int, int, int, str]]
    ) -> dict[str, list | int | str]:
        is_mbart = "mbart" in self.tokenizer.name_or_path.lower()
        is_t5 = "t5" in self.tokenizer.name_or_path.lower() or "t0" in self.tokenizer.name_or_path.lower()
        is_bart = "bart" in self.tokenizer.name_or_path.lower() and not is_mbart
        is_custom_seq2seq = "bert" in self.tokenizer.name_or_path.lower() or "gpt" in self.tokenizer.name_or_path.lower()
        is_bert = "bert" in self.tokenizer.name_or_path.lower()

        # Tokenize each word separately
        tokens = []
        word_mapping = []
        for i, word in enumerate(sentence_raw_words):
            word = " " + word if i > 0 else word
            word_tokens = self.tokenizer.tokenize(word)
            word_mapping.append(list(range(len(tokens), len(tokens) + len(word_tokens))))
            tokens.extend(word_tokens)

        # Convert tokens to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Add special tokens
        if is_mbart:
            input_encoding_ids = [self.tokenizer.lang_code_to_id[self.language]] + input_ids + [self.tokenizer.eos_token_id]  # type: ignore
        elif is_bart:
            input_encoding_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]  # type: ignore
        elif is_t5:
            input_encoding_ids = input_ids + [self.tokenizer.eos_token_id]  # type: ignore
        elif is_bert:
            input_encoding_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]  # type: ignore
        elif is_custom_seq2seq:
            input_encoding_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]  # type: ignore
        else:
            raise ValueError(f"Unsupported tokenizer: {self.tokenizer.name_or_path}")

        encoder_attention_mask = [1] * len(input_encoding_ids)

        # Adjust word_mapping for special tokens at the beginning
        offset = 1 if not is_t5 else 0
        word_mapping = [[i + offset for i in indices] for indices in word_mapping]

        # Create pointer labels and decoder input
        pointer_labels = []
        if is_mbart:
            output_encoding = [self.tokenizer.lang_code_to_id[self.language]]  # type: ignore
        elif is_bart:
            output_encoding = [self.model.config.decoder_start_token_id]
        elif is_t5:
            output_encoding = [self.model.config.eos_token_id]
        elif is_bert:
            output_encoding = [self.tokenizer.cls_token_id]
        elif is_custom_seq2seq:
            output_encoding = [self.tokenizer.bos_token_id]
        else:
            output_encoding = []

        offset = 1 + self.num_labels

        pointer_labels, output_encoding = self.pointer_generator(labels, offset, word_mapping, input_encoding_ids, output_encoding)

        pointer_labels.append(0)  # EOS token points to 0
        decoder_attention_mask = [1] * len(output_encoding)

        max_label = max(pointer_labels)
        assert max_label < len(input_encoding_ids) + self.num_labels, f"Label {max_label} out of range"

        return {
            "encoder_input_ids": input_encoding_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_input_ids": output_encoding,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": pointer_labels,
            "original_sentence": sentence,
            "original_labels": labels,
            "sentence_raw_words": sentence_raw_words,
            "word_mapping": word_mapping,
        }
