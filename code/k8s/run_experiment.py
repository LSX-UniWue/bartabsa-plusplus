import argparse
import logging
import os
import subprocess
import uuid
from datetime import datetime
from typing import Optional, Set

import yaml
from jinja2 import Environment, FileSystemLoader

# Configuration
SSH_SERVER = "your_user@your_server"  # Replace with your SSH server
USER_DIR = "/home/your_user"  # Replace with your home directory
NAMESPACE = "your_namespace"  # Replace with your Kubernetes namespace
TEMP_RUNS_PATH = f"{USER_DIR}/projects/temp_runs"
LOCAL_REPO_ROOT = "/path/to/your/local/repo/"  # Replace with your local repository path
TEMPLATE_FILE = "job_template.yaml.jinja2"
PROJECT_NAME = "BARTSA-Lightning-Experiments"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_ssh_command(command: str) -> subprocess.CompletedProcess:
    full_command = f"ssh {SSH_SERVER} '{command}'"
    return subprocess.run(full_command, shell=True, check=True, capture_output=True, text=True)


def create_temp_folder() -> str:
    temp_folder = f"temp_run_{uuid.uuid4().hex}"
    folder_path = f"{TEMP_RUNS_PATH}/{temp_folder}"
    run_ssh_command(f"mkdir -p {folder_path}")
    logger.info(f"Temp folder created: {folder_path}")
    return temp_folder


def copy_code_to_temp_folder(temp_folder: str) -> None:
    source_path = f"{USER_DIR}/projects/bartabsa-lightning-base/*"
    destination_path = f"{TEMP_RUNS_PATH}/{temp_folder}/"
    run_ssh_command(f"cp -r {source_path} {destination_path}")
    run_ssh_command(f"mkdir -p {destination_path}checkpoints")
    logger.info(f"Copied code to temp folder: {temp_folder}")


def commit_run(job_name: str) -> None:
    subprocess.run(["git", "add", LOCAL_REPO_ROOT])
    subprocess.run(["git", "commit", "-m", f"Experiment run: {job_name}"])
    subprocess.run(["git", "push"])
    logger.info(f"Committed run: {job_name}")


def get_active_jobs() -> Set[str]:
    result = subprocess.run(["kubectl", "get", "jobs", "-n", NAMESPACE, "-o", "yaml"], capture_output=True, text=True, check=True)
    jobs_yaml = yaml.safe_load(result.stdout)

    active_jobs = set()
    for job in jobs_yaml["items"]:
        if "completionTime" not in job["status"]:
            containers = job["spec"]["template"]["spec"]["containers"]
            for container in containers:
                if "volumeMounts" in container:
                    for volume_mount in container["volumeMounts"]:
                        if volume_mount["mountPath"] == "/bartabsa-lightning":
                            subpath = volume_mount["subPath"]
                            if subpath.startswith("projects/temp_runs/"):
                                temp_folder = subpath.split("/")[-1]
                                active_jobs.add(temp_folder)
    return active_jobs


def get_temp_folders() -> Set[str]:
    result = run_ssh_command(f"ls -1 {TEMP_RUNS_PATH}")
    return set(result.stdout.strip().split())


def clean_unused_temp_folders() -> None:
    active_jobs = get_active_jobs()
    all_temp_folders = get_temp_folders()

    logger.info(f"Active jobs: {active_jobs}")
    logger.info(f"All temp folders: {all_temp_folders} ({len(all_temp_folders)})")

    folders_to_remove = all_temp_folders - active_jobs
    if folders_to_remove:
        remove_command = "; ".join([f"rm -rf {TEMP_RUNS_PATH}/{folder}" for folder in folders_to_remove])
        run_ssh_command(remove_command)
        logger.info(f"Cleaned up {len(folders_to_remove)} unused temp folders")
    else:
        logger.info("No unused temp folders to clean up")


def high_prio_run_exists() -> bool:
    cmd_running = [
        "kubectl",
        "get",
        "pods",
        "--field-selector=status.phase=Running",
        "-o",
        'jsonpath=\'{range .items[?(@.spec.priorityClassName=="research-high")]}{.metadata.name}{"\\n"}{end}\'',
    ]

    cmd_pending = [
        "kubectl",
        "get",
        "pods",
        "--field-selector=status.phase=Pending",
        "-o",
        'jsonpath=\'{range .items[?(@.spec.priorityClassName=="research-high")]}{.metadata.name}{"\\n"}{end}\'',
    ]

    try:
        result_running = subprocess.run(cmd_running, capture_output=True, text=True, check=True)
        result_pending = subprocess.run(cmd_pending, capture_output=True, text=True, check=True)
        output_running = result_running.stdout.strip("'")
        output_pending = result_pending.stdout.strip("'")
        return bool(output_running or output_pending)
    except subprocess.CalledProcessError:
        logger.error("Error executing kubectl command")
        return False


def run_kubernetes_job(temp_folder: str, run_name: str, gpu_type: str, config_file: Optional[str], extra_args: Optional[str]) -> None:
    logger.info(f"Starting Kubernetes job: {run_name}")

    job_args = ""

    # Config file needs to be specified first, as other arguments may override it!
    if config_file:
        job_args += f" --config-name {config_file}"

    job_args += f" experiment.run_name={run_name} experiment.project_name={PROJECT_NAME}"
    if extra_args:
        job_args += f" {extra_args}"

    script_path = os.path.dirname(os.path.abspath(__file__))
    template = Environment(loader=FileSystemLoader(script_path)).get_template(TEMPLATE_FILE)
    timestamped_run_name = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}".lower().replace("_", "-")

    config = {
        "name": timestamped_run_name,
        "config": job_args,
        "run_name": "absa",
        "iterations": 1,
        "priority_class": "research-high" if not high_prio_run_exists() else "research-med",
        "temp_folder": f"projects/temp_runs/{temp_folder}",
        "use_node_selector": False,
        "gpu_type": gpu_type,
        "save_checkpoints": False,
    }

    output_text = template.render(**config)
    logger.info(f"Creating job: {config['name']}")
    command = f"kubectl -n {NAMESPACE} create -f -"
    subprocess.run(command.split(), input=output_text.encode(), check=True)
    logger.info(f"Job {run_name} started. You can check its status using:")
    logger.info(f"kubectl get job {timestamped_run_name} -n {NAMESPACE}")
    logger.info(f"To delete the job, use:")
    logger.info(f"----------------\nkubectl delete job {timestamped_run_name} -n {NAMESPACE}\n----------------")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment on Kubernetes cluster")
    parser.add_argument("--run_name", help="Specify a custom run name")
    parser.add_argument("--config", help="Name of the configuration file")
    parser.add_argument(
        "--gpu_type",
        default="gtx1080ti",
        help="Specify GPU type (default: gtx1080ti; available: a100, rtx4090, rtx2080ti, gtx1080ti, titan, rtx8000)",
    )
    parser.add_argument("--extra_args", help="Additional arguments to pass to the run script")
    parser.add_argument("--clean_runs", action="store_true", help="Only clean unused temp folders without starting a new job")
    parser.add_argument("--commit", action="store_true", help="Commit the run to the repository")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.clean_runs:
        clean_unused_temp_folders()

    run_name = args.run_name or f"experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    temp_folder = create_temp_folder()
    copy_code_to_temp_folder(temp_folder)

    run_kubernetes_job(temp_folder, run_name, args.gpu_type, args.config, args.extra_args)

    if args.commit:
        commit_run(run_name)


if __name__ == "__main__":
    main()
