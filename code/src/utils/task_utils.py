import argparse
import time
from logging import getLogger

from omegaconf import OmegaConf
from requests.exceptions import RequestException
from src.utils.config_types import AbsaConfig, compare_configs
from wandb import Api
from wuenlp.impl.UIMANLPStructs import UIMASentimentTuple

logger = getLogger("lightning.pytorch")


def cmp_aspect(v1: UIMASentimentTuple, v2: UIMASentimentTuple):
    """Used to sort sentiment tuples according to expression, or if expression is the same, according to target begin"""
    if v1.expression:
        if v1.expression.token_begin_within(v1) == v2.expression.token_begin_within(v2):
            return v1.target.token_begin_within(v1) - v2.target.token_begin_within(v2)
        return v1.expression.token_begin_within(v1) - v2.expression.token_begin_within(v2)
    else:
        if v1.source.token_begin_within(v1) == v2.source.token_begin_within(v2):
            return v1.target.token_begin_within(v1) - v2.target.token_begin_within(v2)
        return v1.source.token_begin_within(v1) - v2.source.token_begin_within(v2)


def cmp_aspect_entity(v1: UIMASentimentTuple, v2: UIMASentimentTuple):
    if v1.begin == v2.begin:
        return v1.end - v2.end
    return v1.begin - v2.begin


def generate_wandb_run_name(run_config: AbsaConfig) -> str:
    """
    Generates a run name for a wandb run based on the provided configuration.
    The run name is constructed from the dataset name and the changed parameters compared to the default configuration.
    """
    if run_config.experiment.run_name != "default":
        logger.info(f"Using custom run name: {run_config.experiment.run_name}")
        return run_config.experiment.run_name

    default_cfg = OmegaConf.load("conf/config.yaml")  # type: ignore
    default_cfg.pop("defaults", None)  # type: ignore
    default_cfg = OmegaConf.merge(OmegaConf.structured(AbsaConfig), default_cfg)
    changed_params = compare_configs(default_cfg, run_config)  # type: ignore
    run_name = run_config.dataset.name or run_config.dataset.task
    if not changed_params:
        logger.info("No parameters changed, using default run name.")
        return run_name + "_default"
    else:
        logger.info(f"Changed parameters: {changed_params}")
        changed_params.pop("dataset.name", None)
        # Directories are not needed in the run name
        changed_params = {
            k: v for k, v in changed_params.items() if not k.startswith("directories") and not k.startswith("dataset.special_tokens_config")
        }
    for key, value in changed_params.items():
        run_name += f"_{key}={value}"
    return run_name


def check_run_exists(project, run_name, max_retries=3, retry_delay=5, timeout=30):
    """
    Check if a run with the given name exists and is finished in the specified project.
    The wandb API is sometimes flaky, so we retry a few times.
    """
    for attempt in range(max_retries):
        try:
            api = Api(timeout=timeout)
            runs = api.runs(project)
            return any(run.name == run_name and run.state == "finished" for run in runs)
        except RequestException as e:
            logger.warning(f"Network error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Unable to check run existence.")
                raise e
        except ValueError as e:
            # This usually happens when the project is not found, which is fine
            logger.error(f"Unexpected error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise e
    return False
