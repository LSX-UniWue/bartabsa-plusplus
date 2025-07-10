import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartModel,
    EncoderDecoderModel,
    MBartForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    T5Model,
)
from transformers.modeling_utils import get_parameter_dtype

logger = logging.getLogger(__name__)


def generate_tokenizer_with_special_tokens(tokenizer_name: str, special_tokens_mapping: dict[str, str]) -> dict[str, Any]:
    # If the tokenizer_name is a tuple (i.e. we use a custom Seq2Seq model), we use the encoder tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        add_prefix_space=True,
        clean_up_tokenization_spaces=False,
    )
    original_num_tokens = len(base_tokenizer)
    base_vocab = base_tokenizer.get_vocab()
    tokens_to_add = list(special_tokens_mapping.values())

    for token in tokens_to_add:
        assert token not in base_vocab, f"Token {token} already exists in the tokenizer's vocabulary."

    amended_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        additional_special_tokens=tokens_to_add,
        use_fast=True,
        split_special_tokens=False,
        add_prefix_space=True,
        clean_up_tokenization_spaces=False,
    )

    mapping2id = {}
    mapping2targetid = {}

    for key, value in special_tokens_mapping.items():
        key_ids = amended_tokenizer.convert_tokens_to_ids(amended_tokenizer.tokenize(value))
        assert type(key_ids) == list and len(key_ids) == 1, f"Special token {value} is not mapped to a single token: {key_ids}"
        key_id = key_ids[0]

        mapping2id[key] = key_id
        mapping2targetid[key] = len(mapping2targetid)

    label_ids = list(mapping2id.values())

    return {
        "tokenizer": amended_tokenizer,
        "mapping2id": mapping2id,
        "mapping2targetid": mapping2targetid,
        "label_ids": label_ids,
        "original_num_tokens": original_num_tokens,
    }


def embed_special_tokens(
    model: BartModel | T5Model,
    original_num_tokens: int,
    amended_tokenizer: PreTrainedTokenizerFast,
    added_special_tokens: list[str],
    special_tokens_begl: str,
    special_tokens_endl: str,
):
    for token in added_special_tokens:
        token_ids = amended_tokenizer.convert_tokens_to_ids(amended_tokenizer.tokenize(token))
        assert type(token_ids) == list and len(token_ids) == 1, f"Special token {token} is not mapped to a single token: {token_ids}"
        special_token_id = token_ids[0]
        assert special_token_id >= original_num_tokens, f"Special token {token} already existed in the model's vocabulary."

        inner_token = token[len(special_tokens_begl) : -len(special_tokens_endl)]
        inner_token_ids = amended_tokenizer.convert_tokens_to_ids(amended_tokenizer.tokenize(inner_token))
        embeddings = model.get_input_embeddings()(torch.LongTensor(inner_token_ids))
        averaged_embedding = torch.mean(embeddings, dim=0, keepdim=True)
        model.get_input_embeddings().weight.data[special_token_id].copy_(averaged_embedding.squeeze())

    return model


def randomize_weights(model: PreTrainedModel, randomize_encoder: bool, randomize_decoder: bool) -> PreTrainedModel:
    if not (randomize_encoder or randomize_decoder):
        return model

    logger.info(f"Randomizing weights: encoder={randomize_encoder}, decoder={randomize_decoder}")

    # Handle BART and mBART models
    if hasattr(model, "model") and hasattr(model.model, "encoder") and hasattr(model.model, "decoder"):
        logger.info("Detected BART/mBART-like model architecture")
        # For BART-like models, the easiest approach is to create a new model with the same config and then copy just the parts we want to keep
        model_class = type(model)
        config = model.config

        # Step 1: Create a new model instance with the same config but random weights
        logger.info(f"Creating new {model_class.__name__} with random weights")
        new_model = model_class(config)

        # Step 2: Copy over the components we want to keep
        if not randomize_encoder:
            logger.info("Keeping original encoder weights")
            new_model.model.encoder = model.model.encoder

        if not randomize_decoder:
            logger.info("Keeping original decoder weights")
            new_model.model.decoder = model.model.decoder

        # Step 3: Handle tied weights
        # We never want to randomize the shared embeddings, as retraining them is out of scope for our randomization
        logger.info("Copying encoder embeddings to shared and re-tying embeddings")
        new_model.model.shared = model.model.encoder.embed_tokens
        new_model.model.decoder.embed_tokens = new_model.model.encoder.embed_tokens
        model = new_model
        logger.info("Successfully created model with randomized weights")

    # Handle EncoderDecoderModel (for BERT, RoBERTa, GPT-2, etc.) - need a different approach
    elif isinstance(model, EncoderDecoderModel):
        logger.info("Detected custom Seq2Seq (EncoderDecoderModel) architecture")
        if randomize_encoder:
            logger.info("Randomizing encoder weights")
            if model.encoder is not None and hasattr(model.encoder, "config"):
                encoder_config = model.encoder.config
                temp_model = AutoModel.from_config(encoder_config)
                # Tie back the embeddings
                if hasattr(model.encoder, "embeddings"):
                    temp_model.embeddings = model.encoder.embeddings
                elif hasattr(model.encoder, "wte"):
                    temp_model.wte = model.encoder.wte
                else:
                    raise ValueError("Cannot randomize encoder: Could not find embeddings in encoder")
                model.encoder = temp_model
            else:
                logger.warning("Cannot randomize encoder: encoder is None or has no config")

        if randomize_decoder:
            logger.info("Randomizing decoder weights")
            if model.decoder is not None and hasattr(model.decoder, "config"):
                decoder_config = model.decoder.config
                temp_model = AutoModelForCausalLM.from_config(decoder_config)
                model.decoder = temp_model
            else:
                logger.warning("Cannot randomize decoder: decoder is None or has no config")
    else:
        raise ValueError("Unsupported model architecture for randomization")

    return model
