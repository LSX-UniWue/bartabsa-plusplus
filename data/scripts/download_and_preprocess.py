import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Constants
GITHUB_REPO_URL = "https://github.com/yhcc/BARTABSA.git"
ASTEV2_REPO_URL = "https://github.com/xuuuluuu/SemEval-Triplet-data.git"
REPO_DIR = "data/pengb"
ASTEV2_DIR = "ASTE-Data-V2-EMNLP2020"
DATASETS = ["14lap", "14res", "15res", "16res"]


def download_and_extract(repo_url: str, repo_dir: str, extract_path: Path):
    logger.info(f"Cloning repository from {repo_url}")
    with tempfile.TemporaryDirectory() as temp_repo_dir:
        # Clone the repository
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_repo_dir], check=True)

        logger.info(f"Creating archive of {repo_dir}")
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temp_file:
            # Create archive of the specific directory
            subprocess.run(["git", "archive", "--format", "tar", "--output", temp_file.name, "HEAD", repo_dir], cwd=temp_repo_dir, check=True)

        logger.info(f"Extracting data to {extract_path}")
        shutil.unpack_archive(temp_file.name, extract_path)
        os.unlink(temp_file.name)


def download_and_extract_astev2(repo_url: str, extract_path: Path):
    logger.info(f"Cloning ASTE-V2 repository from {repo_url}")
    with tempfile.TemporaryDirectory() as temp_repo_dir:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_repo_dir], check=True)

        source_dir = Path(temp_repo_dir) / ASTEV2_DIR
        for dataset in DATASETS:
            dataset_dir = extract_path / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)

            for file_name in ["train_triplets.txt", "dev_triplets.txt", "test_triplets.txt"]:
                source_file = source_dir / dataset / file_name
                if source_file.exists():
                    new_name = "valid.txt" if file_name == "dev_triplets.txt" else file_name.replace("_triplets.txt", ".txt")
                    shutil.copy(source_file, dataset_dir / new_name)
                    logger.info(f"Copied {source_file} to {dataset_dir / new_name}")


def find_data_dir(base_dir: Path) -> Path:
    for root, dirs, files in os.walk(base_dir):
        if any(dataset in dirs for dataset in DATASETS):
            return Path(root)
    raise FileNotFoundError(f"Could not find data directory containing datasets in {base_dir}")


def process_files(input_folder: Path, xmi_output_folder: Path, json_output_folder: Path):
    logger.info("Processing JSON files and converting to XMI format")

    for dataset in DATASETS:
        dataset_input_dir = input_folder / dataset
        dataset_json_dir = json_output_folder / dataset

        dataset_json_dir.mkdir(parents=True, exist_ok=True)

        if dataset_input_dir.exists():
            # Copy JSON files
            for json_file in dataset_input_dir.glob("*.json"):
                new_json_name = json_file.name.replace("_convert.json", ".json")
                new_json_name = "valid.json" if new_json_name == "dev.json" else new_json_name
                shutil.copy(json_file, dataset_json_dir / new_json_name)
                logger.info(f"Copied {json_file} to {dataset_json_dir / new_json_name}")

    # Convert JSON to XMI
    logger.info("Converting JSON files to XMI format")
    # Get directory of current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    command = [sys.executable, os.path.join(current_dir, "json_to_xmi.py"), "-in", str(json_output_folder), "-out", str(xmi_output_folder)]
    subprocess.run(command, check=True)
    logger.info(f"Converted JSON files in {json_output_folder} to XMI files in {xmi_output_folder}")


def main():
    logger.info("Starting download and preprocessing of ABSA data")

    # Create temporary and output directories
    temp_dir = Path(tempfile.mkdtemp())
    # Get the absolute path of the script's directory
    script_dir = Path(__file__).parent.resolve()
    xmi_output_dir = script_dir / "../pengb/xmi"
    json_output_dir = script_dir / "../pengb/json"
    astev2_output_dir = script_dir / "../astev2"
    xmi_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)
    astev2_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download and extract data
        download_and_extract(GITHUB_REPO_URL, REPO_DIR, temp_dir)

        # Find the correct data directory
        data_dir = find_data_dir(temp_dir)
        logger.info(f"Found data directory: {data_dir}")

        # Process JSON files and convert to XMI
        process_files(data_dir, xmi_output_dir, json_output_dir)

        # Download and extract ASTE-V2 data
        download_and_extract_astev2(ASTEV2_REPO_URL, astev2_output_dir)

        logger.success("Data download and preprocessing completed successfully")
        logger.info(f"Preprocessed XMI data is available in {xmi_output_dir}")
        logger.info(f"Original JSON data is available in {json_output_dir}")
        logger.info(f"ASTE-V2 data is available in {astev2_output_dir}")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
