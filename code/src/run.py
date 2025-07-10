import os
import sys

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set the working directory to the project root to make relative paths work correctly
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import logging

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from src.dataset.data_module import AbsaDataModule
from src.model.module import AbsaEncoderDecoderModule
from src.utils.config_types import AbsaConfig
from src.utils.data_utils import import_special_tokens_mapping
from src.utils.task_utils import check_run_exists, generate_wandb_run_name

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
logger = logging.getLogger("lightning.pytorch")
logger.setLevel(logging.INFO)

cs = ConfigStore.instance()
cs.store(name="base_config", node=AbsaConfig)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(absa_config: AbsaConfig) -> None:
    """
    Runs the training loop with a subsequent evaluation, given some hyperparameters.

    Args:
        hparams (Namespace): The namespace object containing the hyperparameters.
    """
    # Seed everything we can get our hands on
    L.seed_everything(absa_config.experiment.seed, workers=True)
    logger.info(f"Seeding with {absa_config.experiment.seed}")

    # If the run already exists, we skip it (useful for cluster runs with multiple iterations in one job)
    if not absa_config.experiment.ignore_existing and check_run_exists(absa_config.experiment.project_name, generate_wandb_run_name(absa_config)):
        logger.info("Run already exists, skipping...")
        return

    output_dir = HydraConfig().get().runtime.output_dir
    logs_dir = os.path.join(output_dir, absa_config.directories.logs)
    if not absa_config.directories.checkpoints.startswith("/"):
        checkpoints_dir = os.path.join(output_dir, absa_config.directories.checkpoints)
    else:
        checkpoints_dir = absa_config.directories.checkpoints
    os.makedirs(logs_dir, exist_ok=True)  # wandb doesn't like to create the directory itself...
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Assert that the special tokens mapping is set correctly
    assert (
        absa_config.dataset.special_tokens_config is None
    ), "Special tokens mapping should be set automatically, please remove the special_tokens_mapping field from the config."
    absa_config.dataset.special_tokens_config = import_special_tokens_mapping(
        absa_config.dataset.task, absa_config.dataset.special_tokens_file, absa_config.directories.special_tokens_mappings
    )

    logger.info("Starting the training process...")
    logger.info("-----------------------------------")
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory  : {output_dir}")
    logger.info(f"Config directory  : {HydraConfig().get().runtime.cwd}")
    logger.info(f"Logs directory    : {logs_dir}")
    logger.info(f"Checkpoints dir   : {checkpoints_dir}")
    logger.info("Running with hyperparameters:")
    logger.info(f"{absa_config}")
    logger.info("-----------------------------------")

    wandblogger = WandbLogger(
        name=generate_wandb_run_name(absa_config),
        save_dir=logs_dir,
        project=absa_config.experiment.project_name,
        offline=absa_config.experiment.offline,
        log_model=False,  # Change this to log checkpoints (if needed)
        entity=absa_config.experiment.entity,
    )

    # Save the hyperparameters to wandb to allow for easy filtering
    wandblogger.experiment.config.update(OmegaConf.to_container(absa_config, resolve=True, throw_on_missing=True))

    datamodule = AbsaDataModule(absa_config)
    absa_config.dataset.special_tokens_config.mapping2targetID = datamodule.mapping2targetID
    absa_config.dataset.special_tokens_config.mapping2ID = datamodule.mapping2ID

    module = AbsaEncoderDecoderModule(
        absa_config,
        logs_dir,
    )

    best_triplet_checkpoint = ModelCheckpoint(
        monitor=absa_config.monitoring.metric,
        mode="max",
        save_top_k=absa_config.monitoring.save_top_k,
        dirpath=checkpoints_dir,
        filename="{hparams.monitor_metric}-{epoch:02d}-{hparams.monitor_metric:.2f}",
        save_last=True,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="1",
        logger=wandblogger,
        callbacks=[
            EarlyStopping(monitor=absa_config.monitoring.metric, patience=absa_config.training.early_stop_patience, mode="max", strict=True),
            best_triplet_checkpoint,
        ],
        fast_dev_run=absa_config.experiment.debug,
        max_epochs=absa_config.training.max_epochs,
        default_root_dir=logs_dir,
        gradient_clip_val=absa_config.optimizer.gradient_clip.value,
        gradient_clip_algorithm=absa_config.optimizer.gradient_clip.algorithm,
        precision=absa_config.training.precision,  # type: ignore
        accumulate_grad_batches=absa_config.training.accumulate_grad_batches,
        enable_checkpointing=True,
        log_every_n_steps=absa_config.training.log_every_n_steps,
        check_val_every_n_epoch=absa_config.training.check_val_every_n_epoch,
        deterministic=absa_config.experiment.deterministic,
    )

    # Train the model, that's literally it, Joe we did it
    trainer.fit(module, datamodule=datamodule)

    # Test the model (to find out if Joe, indeed, did it)
    if not absa_config.experiment.debug:
        checkpoint = best_triplet_checkpoint.last_model_path if absa_config.training.use_last_checkpoint else best_triplet_checkpoint.best_model_path
        module = AbsaEncoderDecoderModule.load_from_checkpoint(checkpoint)
        trainer.test(module, datamodule=datamodule)

    # Explicitly finish the run to avoid errors with multiple runs in the same process
    wandblogger.experiment.finish()


if __name__ == "__main__":
    # See https://github.com/pytorch/pytorch/issues/11201 (if "too many open files" errors occurs)
    # torch.multiprocessing.set_sharing_strategy("file_system")
    main()
