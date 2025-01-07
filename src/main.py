import argparse
from contextlib import contextmanager
from copy import deepcopy
import logging
import yaml
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List
import threading

from runner.run_manager import RunManager
from wisdom.core.config.config import Override
from wisdom.experimental.CHESS.src.threading_utils import ordered_concurrent_function_calls
from wisdom.platforms.k8s_context import K8sWisdomStaticContext
from wisdom.query_parser.processors.nl_query_processor.nl_query_processor import NLQueryProcessor

# Temporarily add the path for wisdom imports
wisdom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, wisdom_path)

sys.path.pop(0)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline with the specified configuration."
    )
    parser.add_argument(
        "--data_mode", type=str, required=True, help="Mode of the data to be processed."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use.")
    parser.add_argument("--log_level", type=str, default="warning", help="Logging level.")
    parser.add_argument(
        "--pick_final_sql",
        type=bool,
        default=False,
        help="Pick the final SQL from the generated SQLs.",
    )
    parser.add_argument(
        "--wisdom_pipeline",
        action="store_true",
        help="Run the wisdom pipeline instead of CHESS.",
    )
    parser.add_argument(
        "--question_threads", type=int, default=1, help="Parallelization over questions"
    )
    parser.add_argument(
        "--directory_prefix", type=str, default="", help="Prefix for the directory name"
    )
    args = parser.parse_args()

    args.run_start_time = datetime.now().isoformat()
    with open(args.config, "r") as file:
        args.config = yaml.safe_load(file)

    return args


# Thread-local storage to keep track of log files for each thread
thread_local = threading.local()


class ThreadLogHandler(logging.Handler):
    def emit(self, record):
        # Get the log file from thread-local storage
        log_file = getattr(thread_local, "log_file", "default.log")
        # Open the log file and write the log message
        with open(log_file, "a") as f:
            f.write(self.format(record) + "\n")


def setup_logging():
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a handler that uses the log file from the filter
    handler = ThreadLogHandler()
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)


def set_thread_log_file(log_file):
    # Set the log file for the current thread
    thread_local.log_file = log_file


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, "r") as file:
        dataset = json.load(file)
    return dataset


def worker(args, worker_id, nlp_engine, zsheet_store):

    dataset = load_dataset(args.data_path)

    args.worker_id = worker_id
    run_manager = RunManager(args, dataset, nlp_engine, zsheet_store)

    # Set the log file for this worker thread
    set_thread_log_file(f"{run_manager.result_directory}/wisdom.log")

    # Use the root logger
    logging.info(f"Running worker {worker_id}")
    run_manager.run_tasks()
    run_manager.generate_sql_files()
    logging.info(f"Finished worker {worker_id}")


def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()

    override = Override()
    override["query_parser"]["eval_mode"] = "false"
    override["query_cache"]["enabled"] = "false"
    override["query_cache"]["lookup_candidates"] = "false"
    override["query_cache"]["writable"] = "false"
    wisdom_context = K8sWisdomStaticContext(override=override)
    nlp_engine: NLQueryProcessor = NLQueryProcessor(wisdom_context)
    zsheet_store = wisdom_context.zsheet_store()

    # Call setup_logging once at the start of your application
    setup_logging()

    ordered_concurrent_function_calls([
        {
            "function": worker,
            "kwargs": {
                "args": deepcopy(args),
                "worker_id": worker_id,
                "nlp_engine": nlp_engine,
                "zsheet_store": zsheet_store,
            },
        }
        for worker_id in range(args.question_threads)
    ])


if __name__ == "__main__":
    main()
