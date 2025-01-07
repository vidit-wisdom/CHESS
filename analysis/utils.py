from dataclasses import dataclass
import json
from typing import Any
import os

from wisdom.experimental.CHESS.src.database_utils.execution import execute_sql


@dataclass
class BirdExampleID:
    id: int
    db_id: str

    def __str__(self):
        return f"{self.db_id}_{self.id}"

    def __hash__(self):
        return hash(str(self))


@dataclass
class RunStats:
    correct_count: int
    incorrect_count: int
    error_count: int
    total_count: int
    correct_examples: list[BirdExampleID]
    incorrect_examples: list[BirdExampleID]
    error_examples: list[BirdExampleID]

    def print_counts(self):
        print(f"Total: {self.total_count}")
        print(f"Correct: {self.correct_count}")
        print(f"Incorrect: {self.incorrect_count}")
        print(f"Error: {self.error_count}")


def load_stats(run_names: list[str], db_id: str | None = None) -> RunStats:

    correct_examples: list[BirdExampleID] = []
    incorrect_examples: list[BirdExampleID] = []
    error_examples: list[BirdExampleID] = []

    for run_name in run_names:
        dev_path = "python/wisdom/experimental/CHESS/results/dev/CHESS_IR_SS_CG/dev/"
        for folder in os.listdir(dev_path):
            if folder.startswith(run_name):
                run_path = os.path.join(dev_path, folder, "-statistics.json")
                with open(run_path, "r") as f:
                    stats_dict = json.load(f)
                ids_dict = stats_dict["ids"]["final_SQL"]

                correct_examples.extend([
                    BirdExampleID(id=id_list[1], db_id=id_list[0])
                    for id_list in ids_dict["correct"]
                    if id_list[0] == db_id or db_id is None
                ])
                incorrect_examples.extend([
                    BirdExampleID(id=id_list[1], db_id=id_list[0])
                    for id_list in ids_dict["incorrect"]
                    if id_list[0] == db_id or db_id is None
                ])
                error_examples.extend([
                    BirdExampleID(id=id_list[1], db_id=id_list[0])
                    for id_list in ids_dict["error"]
                    if id_list[0] == db_id or db_id is None
                ])

    return RunStats(
        correct_count=len(correct_examples),
        incorrect_count=len(incorrect_examples),
        error_count=len(error_examples),
        total_count=len(correct_examples) + len(incorrect_examples) + len(error_examples),
        correct_examples=correct_examples,
        incorrect_examples=incorrect_examples,
        error_examples=error_examples,
    )


def get_failing_examples(stats: RunStats, baseline_stats: RunStats):
    failing_examples: list[BirdExampleID] = []
    for example in stats.incorrect_examples + stats.error_examples:
        if example in baseline_stats.correct_examples:
            failing_examples.append(example)
    return failing_examples


def execute_sql_on_db(db_id: str, sql: str, drop_duplicates: bool = False) -> Any:
    db_path = (
        f"python/wisdom/experimental/CHESS/data/dev_20240627/dev_databases/{db_id}/{db_id}.sqlite"
    )
    results = execute_sql(db_path, sql)
    return set(results) if drop_duplicates else results
