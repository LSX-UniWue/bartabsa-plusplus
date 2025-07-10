import datetime
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import evaluate
import torch
import transformers
from datasets import Dataset, DatasetDict
from huggingface_hub import HfFolder, login
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EncoderDecoderModel,
    HfArgumentParser,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_type: str = field(default="bert2bert", metadata={"help": "Model type to use: bert2bert, roberta2roberta, gpt22gpt2, bart"})
    model_name_or_path: str = field(
        default="bert-base-uncased", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    decoder_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained decoder model (for encoder-decoder models)"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(default="cnn_dailymail", metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: str = field(default="3.0.0", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    max_source_length: int = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization."})
    max_target_length: int = field(default=128, metadata={"help": "The maximum total sequence length for target text after tokenization."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging or quicker training, truncate the number of training examples to this value if set."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging or quicker training, truncate the number of evaluation examples to this value if set."},
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging or quicker training, truncate the number of test examples to this value if set."},
    )
    num_beams: int = field(
        default=4,
        metadata={"help": "Number of beams to use for evaluation. Used during ``evaluate`` and ``predict``."},
    )
    source_column: str = field(
        default="article",
        metadata={"help": "The name of the column in the datasets containing the input texts."},
    )
    target_column: str = field(
        default="highlights",
        metadata={"help": "The name of the column in the datasets containing the target texts."},
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments for training.
    """

    output_dir: str = field(
        default="./outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "A descriptive name for the run to be used by wandb."},
    )
    debug_thingy: bool = field(default=False, metadata={"help": "Whether to run in debug mode with minimal data and training."})
    push_to_hub: bool = field(default=False, metadata={"help": "Whether or not to push the model to the Hub."})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."})
    hub_token: Optional[str] = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    hub_private_repo: bool = field(default=False, metadata={"help": "Whether to create a private repository when pushing to the Hub."})


def main():
    # Parse arguments
    # Note: Using a list instead of a tuple to avoid type errors with HfArgumentParser
    parser = HfArgumentParser([ModelArguments, DataTrainingArguments, TrainingArguments])  # type: ignore
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set a descriptive run_name if not provided
    if training_args.run_name is None:
        training_args.run_name = f"{model_args.model_type}_{model_args.model_name_or_path}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Detect and set the appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")

    # Log the device being used
    logger.info(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Check if we can push to hub and login if needed
    if training_args.push_to_hub:
        if training_args.hub_token:
            login(token=training_args.hub_token)
        elif os.environ.get("HUGGINGFACE_TOKEN"):
            login(token=os.environ.get("HUGGINGFACE_TOKEN"))
        else:
            logger.warning("No Hugging Face token provided. Make sure you are logged in with `huggingface-cli login`")

        # Test connection to hub
        try:
            if not HfFolder.get_token():
                raise ValueError("Not logged in to Hugging Face Hub. Please provide a token.")
            logger.info("Successfully authenticated with Hugging Face Hub")
        except Exception as e:
            logger.error(f"Error connecting to Hugging Face Hub: {e}")
            logger.error("Training will continue but model won't be pushed to Hub")
            training_args.push_to_hub = False

    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    raw_datasets = datasets.load_dataset(  # type: ignore
        data_args.dataset_name,
        data_args.dataset_config_name,
    )

    # Handle debug mode
    if training_args.debug_thingy:
        logger.info("Running in debug mode")
        # Use very small subset of data
        if isinstance(raw_datasets, DatasetDict):
            for split in raw_datasets:
                max_samples = 32 if split == "train" else 16
                if isinstance(raw_datasets[split], Dataset):
                    raw_datasets[split] = raw_datasets[split].select(range(min(max_samples, len(raw_datasets[split]))))
        else:
            logger.warning("raw_datasets is not a DatasetDict, not truncating data for debug mode")

        # Set minimal training parameters
        training_args.num_train_epochs = 1
        training_args.logging_steps = 1
        training_args.eval_steps = 10
        training_args.save_steps = 10
        training_args.warmup_steps = 5

    # Load tokenizer
    tokenizer_name = model_args.model_name_or_path
    logger.info(f"Loading tokenizer: {tokenizer_name}")

    # Initialize tokenizer based on model type
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=model_args.use_fast_tokenizer)

    # Set special tokens based on model type
    if model_args.model_type == "bert2bert":
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    elif model_args.model_type == "roberta2roberta":
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.cls_token if tokenizer.cls_token else "<s>"
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.sep_token if tokenizer.sep_token else "</s>"
    elif model_args.model_type == "gpt22gpt2":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif model_args.model_type not in ["bart", "t5"]:
        raise ValueError(f"Unsupported model type: {model_args.model_type}")

    # Load model
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    if model_args.model_type in ["bert2bert", "roberta2roberta", "gpt22gpt2"]:
        decoder_name = model_args.decoder_model_name_or_path or model_args.model_name_or_path
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_args.model_name_or_path, decoder_name)

        # Set special tokens
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # Set generation parameters
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.max_length = 142
        model.config.min_length = 56
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = data_args.num_beams
    else:  # bart, t5
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

        # Set generation parameters for BART/T5 as well to ensure consistency across all models
        model.config.max_length = 142
        model.config.min_length = 56
        model.config.no_repeat_ngram_size = 3
        model.config.early_stopping = True
        model.config.length_penalty = 2.0
        model.config.num_beams = data_args.num_beams

    # Move model to the appropriate device
    # Using the string representation of the device to avoid type errors
    model = model.to(device.type)  # type: ignore

    # Preprocessing function
    def preprocess_function(examples):  # type: ignore
        """Process a batch of examples for training/evaluation."""
        inputs = examples[data_args.source_column]
        targets = examples[data_args.target_column]

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True)

        # Tokenize targets
        labels = tokenizer(targets, max_length=data_args.max_target_length, padding="max_length", truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
        # to ignore padding in the loss
        if tokenizer.pad_token_id is not None:
            labels_input_ids = labels["input_ids"]

            # Process each sequence in the batch
            processed_labels = []
            for label_ids in labels_input_ids:  # type: ignore
                # Convert to list if not already
                if not isinstance(label_ids, list):
                    label_ids = list(label_ids)

                # Replace pad tokens with -100
                processed_label = [l if l != tokenizer.pad_token_id else -100 for l in label_ids]
                processed_labels.append(processed_label)

            labels["input_ids"] = processed_labels

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Preprocess the datasets
    train_dataset: datasets.Dataset = raw_datasets["train"]  # type: ignore
    eval_dataset: datasets.Dataset = raw_datasets["validation"]  # type: ignore
    test_dataset: datasets.Dataset = raw_datasets["test"]  # type: ignore

    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on train dataset",
    )

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=eval_dataset.column_names,
        desc="Running tokenizer on validation dataset",
    )

    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=test_dataset.column_names,
        desc="Running tokenizer on test dataset",
    )

    # Load ROUGE for evaluation
    rouge = evaluate.load("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # For Seq2SeqTrainer with predict_with_generate=True
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Handle case where predictions are not token ids but logits
        if len(pred_ids.shape) == 3:
            # Convert logits to token ids by taking argmax
            pred_ids = pred_ids.argmax(axis=-1)

        # Replace -100 with pad token id in labels
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Compute ROUGE scores
        try:
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

            # Check if result is None or if any of the metrics are None
            if result is None:
                logger.warning("ROUGE computation returned None. Using default values.")
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

            # Extract metrics with fallback to 0.0 if any are None
            rouge1 = result.get("rouge1", 0.0)
            rouge2 = result.get("rouge2", 0.0)
            rougeL = result.get("rougeL", 0.0)
            rougeLsum = result.get("rougeLsum", 0.0)

            return {
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rougeL,
                "rougeLsum": rougeLsum,
            }
        except Exception as e:
            logger.error(f"Error in compute_metrics: {e}")
            # Return empty metrics in case of error
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training")
    train_result = trainer.train()

    # Save the model
    logger.info("Saving final model")
    trainer.save_model()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_results = trainer.predict(test_dataset)  # type: ignore
    test_metrics = test_results.metrics

    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # Push to hub if requested
    if training_args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub")
        trainer.push_to_hub()

    return test_metrics


if __name__ == "__main__":
    main()
