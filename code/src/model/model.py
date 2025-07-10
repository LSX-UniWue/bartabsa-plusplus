import math
from logging import getLogger
from typing import Any, Optional

import torch
from src.utils.config_types import AbsaConfig
from src.utils.model_utils import embed_special_tokens, generate_tokenizer_with_special_tokens, randomize_weights
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModel,
    BartForConditionalGeneration,
    BartModel,
    EncoderDecoderModel,
    MBartForConditionalGeneration,
    PreTrainedTokenizerFast,
    T5Model,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bart.modeling_bart import BartAttention, BartForConditionalGeneration, Seq2SeqLMOutput

logger = getLogger("lightning.pytorch")


class GatingMechanism(nn.Module):
    """
    Super simple gating mechanism, allowing for weighting per token.
    """

    def __init__(self, input_size):
        super().__init__()
        self.gate = nn.Linear(input_size * 2, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=-1)
        gate = self.sigmoid(self.gate(combined))
        gated_value = gate * x + (1 - gate) * y
        return gated_value, gate


class RMSNorm(nn.Module):
    """
    RMSNorm taken from HuggingFace Llama implementation
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AbsaEncoderDecoderModel(torch.nn.Module):
    def __init__(self, config: AbsaConfig):
        super().__init__()

        # Validate configuration again, for type checking
        assert config.dataset.special_tokens_config is not None, "Special tokens mapping is required for the model."
        if config.dataset.lang_code is None and "mbart" in config.model.base_model.lower():
            raise ValueError("Please provide a language code for mbart models.")

        assert config.model.gating_mode in ["full_gating", "encoder_gating", "decoder_gating", "no_gating"], "Invalid gating mode."
        assert config.model.attention_mechanism in ["none", "custom", "bart"], "Invalid attention mechanism."

        # Initialize attributes needed throughout the model
        self.alpha = config.model.alpha
        self.language = config.dataset.lang_code
        self.use_encoder_mlp = config.model.use_encoder_mlp
        self.attention_mechanism = config.model.attention_mechanism
        self.normalize_encoder_outputs = config.model.normalize_encoder_outputs
        self.use_final_layer_norm = config.model.use_final_layer_norm
        self.gating_mode = config.model.gating_mode
        self.use_rms_for_encoder_norm = config.model.use_rms_for_encoder_norm
        self.dont_use_rms = config.model.dont_use_rms
        self.use_dimension_normalization = config.model.use_dimension_normalization
        self.use_value_matrix = config.model.use_value_matrix

        # Initialize tokenizer
        tokenizer_data = generate_tokenizer_with_special_tokens(config.model.base_model, config.dataset.special_tokens_config.special_tokens_mapping)
        self.tokenizer: PreTrainedTokenizerFast = tokenizer_data["tokenizer"]

        # Determine model type
        self.is_custom_seq2seq = config.model.decoder_model is not None and config.model.decoder_model != ""
        self.is_gpt_encoder = "gpt" in config.model.base_model.lower()
        self.is_mbart = "mbart" in self.tokenizer.name_or_path.lower()
        self.is_t5 = "t5" in self.tokenizer.name_or_path.lower() or "t0" in self.tokenizer.name_or_path.lower()
        self.is_bart = "bart" in self.tokenizer.name_or_path.lower() and not self.is_mbart

        # Initialize model
        logger.info(f"Initializing model with dropout {config.model.dropout.general:.2f} and attention dropout {config.model.dropout.attention:.2f}")

        model_params = {}
        if self.is_bart or self.is_mbart:
            model_params = {
                "dropout": config.model.dropout.general,
                "attention_dropout": config.model.dropout.attention,
            }
        elif not self.is_custom_seq2seq:
            raise NotImplementedError(f"Invalid model name: {config.model.base_model} (T5 is not supported in this model-variations for now.)")

        # Initialize the base model
        # model: BartModel | T5Model = AutoModel.from_pretrained(config.model.base_model, **model_params)
        if self.is_custom_seq2seq:
            model: PreTrainedModel = EncoderDecoderModel.from_encoder_decoder_pretrained(
                config.model.base_model, config.model.decoder_model, **model_params
            )
        elif self.is_bart:
            assert isinstance(config.model.base_model, str), "Base model must be a string for BART models"
            model: PreTrainedModel = BartForConditionalGeneration.from_pretrained(config.model.base_model, **model_params)
        else:
            assert isinstance(config.model.base_model, str), "Base model must be a string for mBART models"
            model: PreTrainedModel = MBartForConditionalGeneration.from_pretrained(config.model.base_model, **model_params)

        # Resize token embeddings to account for the new special tokens
        if self.is_custom_seq2seq:
            model.encoder.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
            model.decoder.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
        else:
            model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

        # Embed special tokens
        self.model = embed_special_tokens(
            model,
            tokenizer_data["original_num_tokens"],
            self.tokenizer,
            list(config.dataset.special_tokens_config.special_tokens_mapping.values()),
            config.dataset.special_tokens_config.special_tokens_begl,
            config.dataset.special_tokens_config.special_tokens_endl,
        )

        # Randomize weights if specified in config
        self.model = randomize_weights(self.model, config.model.randomize_encoder, config.model.randomize_decoder)

        # Initialize special tokens mapping
        eos_like_token_id = self.tokenizer.lang_code_to_id[self.language] if self.is_mbart else self.tokenizer.eos_token_id  # type: ignore
        if "bert" in self.tokenizer.name_or_path.lower():
            eos_like_token_id = self.tokenizer.cls_token_id

        self.register_buffer(
            "special_tokens_mapping",
            torch.tensor([eos_like_token_id] + sorted(tokenizer_data["label_ids"]), dtype=torch.long),
        )
        self.src_start_index = len(self.special_tokens_mapping)

        self.model_dim = self.model.config.d_model if not self.is_custom_seq2seq else self.model.config.encoder.hidden_size

        # Initialize encoder MLP if needed
        if self.use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(self.model_dim, self.model_dim),
                nn.ReLU(),
                nn.Dropout(config.model.dropout.encoder_mlp),
                nn.Linear(self.model_dim, self.model_dim),
            )

        # Additional parameters for the gating and attention mechanisms, initialized based on the config
        if self.gating_mode == "encoder_gating":
            self.encoder_gating = GatingMechanism(self.model_dim)
        elif self.gating_mode == "decoder_gating":
            self.attention_gating = GatingMechanism(self.model_dim)
        elif self.gating_mode == "full_gating":
            self.encoder_gating = GatingMechanism(self.model_dim)
            self.attention_gating = GatingMechanism(self.model_dim)
        else:
            self.alpha = self.alpha

        # Initialize attention components
        if self.attention_mechanism == "custom":
            self.W_key = nn.Linear(self.model_dim, self.model_dim, bias=False)
            self.W_query = nn.Linear(self.model_dim, self.model_dim, bias=False)
            if self.use_dimension_normalization:
                self.d = self.model_dim
            if self.use_value_matrix:
                self.W_value = nn.Linear(self.model_dim, self.model_dim, bias=False)
        elif self.attention_mechanism == "bart":
            # Handle different model architectures
            if self.is_bart or self.is_mbart:
                self.decoder_cross_attention = self.model.model.decoder.layers[-1].encoder_attn
                self.cross_attention_type = "bart"
            elif self.is_custom_seq2seq:
                if hasattr(self.model.decoder, "bert"):
                    self.decoder_cross_attention = self.model.decoder.bert.encoder.layer[-1].crossattention
                    self.cross_attention_type = "bert"
                elif hasattr(self.model.decoder, "roberta"):
                    self.decoder_cross_attention = self.model.decoder.roberta.encoder.layer[-1].crossattention
                    self.cross_attention_type = "bert"  # RoBERTa uses the same interface as BERT
                elif hasattr(self.model.decoder, "transformer"):  # The GPT2 case
                    self.decoder_cross_attention = self.model.decoder.transformer.h[-1].crossattention
                    self.cross_attention_type = "gpt2"  # GPT-2 has a different interface
                else:
                    raise ValueError("Wait a second, what is this model architecture?")
            else:
                raise ValueError("Wait a second, what is this model architecture?")

        if self.use_rms_for_encoder_norm:
            self.encoder_norm_concats = RMSNorm(self.model_dim, config.model.rmsnorm_eps)
            self.encoder_norm_hiddens = RMSNorm(self.model_dim, config.model.rmsnorm_eps)
        if self.use_final_layer_norm and not self.dont_use_rms:
            self.final_layer_norm = RMSNorm(self.model_dim, config.model.rmsnorm_eps)
        elif self.use_final_layer_norm and self.dont_use_rms:
            self.final_layer_norm = nn.LayerNorm(self.model_dim, eps=config.model.rmsnorm_eps)

    def forward(self, batch: dict[str, torch.Tensor], generating: bool = False):
        """
        Unified forward method that conditionally applies different architectural components based on configuration.

        Args:
            batch (dict[str, Any]): Batch containing input and decoder data.
            generating (bool, optional): Flag indicating if the model is in generation mode. Defaults to False.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the output logits and intermediate tensors.
        """
        # Ensure the model is BART or mBART
        intermediate_steps = {}
        if not self.is_custom_seq2seq:
            # TODO: Move back to simple one-step forward pass
            # Always encode the input first
            encoder_out = self.model.model.encoder(
                input_ids=batch["encoder_input_ids"],
                attention_mask=batch["encoder_attention_mask"],
            )

            amended_enc_outs = encoder_out.last_hidden_state
            intermediate_steps["enc_outs"] = amended_enc_outs
            amended_enc_att_mask = batch["encoder_attention_mask"]

            decoder_out: Seq2SeqLMOutput = self.model.forward(
                encoder_outputs=[amended_enc_outs],  # type: ignore
                attention_mask=amended_enc_att_mask,  # type: ignore
                decoder_input_ids=batch["decoder_input_ids"],  # type: ignore
                decoder_attention_mask=batch["decoder_attention_mask"],  # type: ignore
                return_dict=True,
                output_attentions=True,
                output_hidden_states=True,
            )
        else:
            decoder_out: Seq2SeqLMOutput = self.model.forward(
                input_ids=batch["encoder_input_ids"],  # type: ignore
                attention_mask=batch["encoder_attention_mask"],  # type: ignore
                decoder_input_ids=batch["decoder_input_ids"],  # type: ignore
                decoder_attention_mask=batch["decoder_attention_mask"],  # type: ignore
                return_dict=True,
                output_attentions=True,
                output_hidden_states=True,
            )  # type: ignore

        # This is the "normal" BARTABSA forward pass (with out other modifications)
        # Going through the calculation from the paper:
        # H^e = BARTEncoder([x_1, x_2, ..., x_n]) (eq. 2)
        H_e: torch.FloatTensor = decoder_out.encoder_last_hidden_state  # (bsz, n, d_model) # type: ignore

        # H(hat)^e = MLP(H^e) (eq. 6)
        H_hat_e = self.encoder_mlp(H_e) if self.use_encoder_mlp else H_e

        # h_t^e = BARTDecoder(H^e, [y_1, y_2, ..., y_T]) (eq. 4)
        # Teacher forcing is used here -> T = len(decoder_input_ids)
        # -> Calculate decoder output for all tokens at once (not just for one like in the papers eqs)
        assert decoder_out.decoder_hidden_states is not None, "Decoder hidden states are not returned by the model"
        H_d = decoder_out.decoder_hidden_states[-1]  # (bsz, T, d_model)

        # E^e = BARTTokenEmbed(X) (eq. 5)
        if not self.is_custom_seq2seq:
            word_embeddings = self.model.model.shared
        elif self.is_gpt_encoder:
            word_embeddings = self.model.encoder.wte
        else:
            word_embeddings = self.model.encoder.embeddings.word_embeddings
        E_e = word_embeddings(batch["encoder_input_ids"])  # (bsz, n, d_model)

        # H(overline)^e = alpha * H(hat)^e + (1 - alpha) * E^e (eq. 7)
        if self.gating_mode == "no_gating" or self.gating_mode == "decoder_gating":
            H_ovl_e = self.alpha * H_hat_e + (1 - self.alpha) * E_e
        elif self.gating_mode == "full_gating" or self.gating_mode == "encoder_gating":
            # Use gating (i.e. learnable weights) instead of alpha
            H_ovl_e, intermediate_steps["encoder_gating_gate"] = self.encoder_gating(H_hat_e, E_e)  # (bsz, n, d_model)
        else:
            raise ValueError(f"Invalid gating mode: {self.gating_mode}")

        # C^d = BARTTokenEmbed(C) (eq. 8)
        # Need to expand C to allow for batch processing
        # l = len(specical_tokens)
        C_d = word_embeddings(self.special_tokens_mapping).unsqueeze(0).expand(batch["encoder_input_ids"].shape[0], -1, -1)  # (bsz, l, d_model)

        # Since H_ovl_e and C_d are differently scaled, we need to normalize them before concatenating
        if self.normalize_encoder_outputs:
            if self.use_rms_for_encoder_norm:
                C_d = self.encoder_norm_concats(C_d)
                H_ovl_e = self.encoder_norm_hiddens(H_ovl_e)
            else:
                H_ovl_e = F.normalize(H_ovl_e, dim=-1)
                C_d = F.normalize(C_d, dim=-1)

        # P_t = softmax(concat(H(overline)^e, C^d) * H_t^e) (eq. 9)
        # The softmax is done in the loss function and omitted here
        # The concat is done the other way around since we add the special tokens in the beginning instead of appending them
        conc_e = torch.cat([C_d, H_ovl_e], dim=1)  # (bsz, n + l, d_model)

        # Attention mechanism
        if self.attention_mechanism != "none":
            if self.attention_mechanism == "bart":
                attention_output = self._apply_cross_attention(conc_e, H_d, batch["head_mask"])
            elif self.attention_mechanism == "custom":
                # Compute keys and queries based on encoder and decoder states, as usual in cross attention
                keys = self.W_key(conc_e)  # (bsz, in_seq_len + 4, hidden_size)
                queries = self.W_query(H_d)  # (bsz, out_seq_len, hidden_size)

                if self.use_value_matrix:
                    values = self.W_value(conc_e)
                else:
                    values = conc_e

                # Compute the attention scores (for now, without scaling)
                attn_scores = torch.einsum("btd,bnd->btn", queries, keys)  # (bsz, out_seq_len, in_seq_len + 4)
                attn_scores = attn_scores.masked_fill(~batch["head_mask"], torch.finfo(attn_scores.dtype).min)
                if self.use_dimension_normalization:
                    attn_scores = attn_scores / math.sqrt(self.d)
                attn_weights = torch.softmax(attn_scores, dim=-1)  # (bsz, out_seq_len, in_seq_len + 4)

                # Use the weights to get a weighted sum of the encoder states
                attention_output = torch.einsum("btn,bnd->btd", attn_weights, values)  # (bsz, out_seq_len, hidden_size)
            else:
                raise ValueError("Invalid attention mechanism")

            if self.gating_mode == "full_gating" or self.gating_mode == "decoder_gating":
                # Apply this attention to the decoder hidden states with gating
                gated_output, intermediate_steps["attention_gating_gate"] = self.attention_gating(
                    attention_output, H_d
                )  # (bsz, seq_len, hidden_size)
            else:
                # If no gating is used, the attention output is already the result
                gated_output = attention_output

            if self.use_final_layer_norm:
                normalized_output = self.final_layer_norm(gated_output)  # (bsz, seq_len, hidden_size)
            else:
                normalized_output = gated_output  # (bsz, seq_len, hidden_size)
        else:
            normalized_output = H_d  # (bsz, seq_len, hidden_size)

        # P_t = torch.einsum("bnd,btd->btn", conc_e, H_d)  # (bsz, n + l, T)
        P_t = torch.einsum("btd,bnd->btn", normalized_output, conc_e)  # (bsz, n + l, T)

        # Lastly we make sure to not attend to padding and special tokens (i.e. the stuff added by the special tokens mapping)
        P_t = P_t.masked_fill(~batch["head_mask"], torch.finfo(P_t.dtype).min)  # (bsz, n + l, T)

        intermediate_steps = {
            "H_e": H_e,
            "H_hat_e": H_hat_e,
            "H_d": normalized_output,
            "E_e": E_e,
            "H_ovl_e": H_ovl_e,
            "C_d": C_d,
            "conc_e": conc_e,
            "P_t": P_t,
        }
        return P_t, intermediate_steps

    def _apply_cross_attention(self, encoder_hidden_states: torch.Tensor, decoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # Prepare inputs
        bsz, tgt_len, _ = decoder_hidden_states.size()
        src_len = encoder_hidden_states.size(1)

        # Determine which cross-attention interface to use
        if self.cross_attention_type == "bert":
            # For BERT-style cross-attention
            # Prepare attention mask for BERT format
            expanded_attn_mask = attention_mask.unsqueeze(1).to(self.decoder_cross_attention.self.query.weight.dtype)
            expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(expanded_attn_mask.dtype).min

            # Call BERT-style cross-attention
            attn_output = self.decoder_cross_attention(
                hidden_states=decoder_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=expanded_attn_mask,
            )

            # If the attention returns a tuple, get the first element
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
        elif self.cross_attention_type == "gpt2":
            # For GPT-2 style cross-attention
            # Prepare attention mask for GPT-2 format
            # For GPT-2, attention_mask should be expanded to [batch_size, 1, 1, seq_length]
            expanded_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_attn_mask = expanded_attn_mask.to(dtype=next(self.decoder_cross_attention.parameters()).dtype)
            expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(expanded_attn_mask.dtype).min

            # Call GPT-2 style cross-attention
            # GPT-2 uses encoder_hidden_states parameter instead of key_value_states
            attn_outputs = self.decoder_cross_attention(
                hidden_states=decoder_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=expanded_attn_mask,
                use_cache=False,
                output_attentions=False,
            )

            # GPT-2's cross-attention returns output as first element in tuple
            attn_output = attn_outputs[0] if isinstance(attn_outputs, tuple) else attn_outputs
        elif self.cross_attention_type == "bart":
            # For BART-style cross-attention (default)
            # Prepare attention mask for BART format
            expanded_attn_mask = attention_mask.unsqueeze(1).to(self.decoder_cross_attention.k_proj.weight.dtype)
            expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(expanded_attn_mask.dtype).min

            # Call BART-style cross-attention
            attn_output, _, _ = self.decoder_cross_attention(
                hidden_states=decoder_hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=expanded_attn_mask,
            )
        else:
            raise ValueError(f"Invalid cross-attention type: {self.cross_attention_type}")

        return attn_output

    @torch.inference_mode()
    def generate(self, batch: dict[str, Any], max_length: int = 50, mode: str = "greedy", beam_size: int = 5):
        """
        Generates a sequence of tokens based on the input sequence.

        Args:
            batch (dict[str, Any]): Batch containing input and decoder data.
            max_length (int, optional): Maximum length of the generated sequence. Defaults to 50.
            mode (str, optional): Generation mode, either "greedy" or "beam". Defaults to "greedy".
            beam_size (int, optional): Beam size for beam search. Defaults to 5.

        Returns:
            torch.Tensor: Generated sequence of tokens.
        """

        assert mode in ["greedy", "beam"], "Only 'greedy' and 'beam' generation modes are supported."
        assert mode == "greedy", "Only 'greedy' generation mode is supported at the moment."

        input_ids = batch["encoder_input_ids"]
        attention_mask = batch["encoder_attention_mask"]
        device = input_ids.device

        decoder_input_ids = batch["decoder_input_ids"]
        decoder_pointer_outputs = torch.zeros(input_ids.size(0), 1, device=device, dtype=torch.long)

        # Since the headmask is already expanded to the final output size, we need to kill that dimension first
        head_mask = batch["head_mask"][:, 0, :].unsqueeze(1)

        is_finished = torch.zeros(input_ids.size(0), device=device).bool()

        for _ in range(max_length):
            P_t, _ = self.forward(
                {
                    "encoder_input_ids": input_ids,
                    "encoder_attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": torch.ones_like(decoder_input_ids, device=device),
                    "head_mask": head_mask.expand(-1, decoder_input_ids.size(1), -1),
                },
                True,
            )

            # Take the logits of the last token
            next_token = P_t[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            decoder_pointer_outputs = torch.cat([decoder_pointer_outputs, next_token], dim=-1)

            # Prepare the next input by de-referencing the last token
            decoder_input_ids = self._map_tokens_to_embeddings(decoder_pointer_outputs, input_ids).to(device)

            # Additionally set the is_finished flag for all sequences that have reached the EOS token
            is_finished = torch.logical_or(is_finished, next_token == 0)

            if torch.all(is_finished):
                break

        return decoder_pointer_outputs[:, 1:]  # Remove the BOS token

    @torch.inference_mode()
    def _map_tokens_to_embeddings(self, decoder_pointer_outputs: torch.Tensor, input_ids: torch.Tensor):
        """
        Maps token IDs to their embeddings, handling both special and normal tokens.

        Args:
            decoder_pointer_outputs (torch.Tensor): Tensor of shape `(bsz, T)` containing token IDs from the decoder output.
            input_ids (torch.Tensor): Tensor of shape `(bsz, n)` containing token IDs from the input sequence.

        Returns:
            torch.Tensor: Tensor containing the mapped token embeddings.
        """
        ######
        # Convert the ids to their actual embeddings (as described in eq. 3 in the paper)
        # Everything before the src_start_index is considered a special token and is mapped to the special tokens ids
        # Everything after the src_start_index is considered part of the input sequence and is mapped from their index to the actual token ids
        ######

        # First, we create a mask to identify the special tokens and revert their indices back to their original token ids
        special_token_mask = decoder_pointer_outputs.lt(self.src_start_index)

        mapped_tokens = decoder_pointer_outputs.masked_fill(decoder_pointer_outputs.ge(self.src_start_index), 0)
        special_mapped_tokens = self.special_tokens_mapping[
            mapped_tokens
        ]  # Mask is necessary since all "non-special" tokens are now mapped to be eos_token_id

        # Now we do the same for the "normal" input tokens
        # Get the "real" index of the tokens first
        src_token_indices = decoder_pointer_outputs - self.src_start_index
        # Replace all special tokens with zeros to avoid issues (masked out later anyway)
        src_token_indices = src_token_indices.masked_fill(src_token_indices.lt(0), 0)

        # Now we map the indices back to the actual token ids (using the encoder's src_tokens, i.e. the original input sequence)
        word_mapped_tokens = input_ids.gather(1, src_token_indices)  # shape: (batch, max_len)

        # We now successfully mapped the special tokens and the input tokens back to their original token ids
        # So lastly, we combine them back together
        dereferenced_prev_output_tokens = torch.where(special_token_mask, special_mapped_tokens, word_mapped_tokens)

        return dereferenced_prev_output_tokens
