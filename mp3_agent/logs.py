"""Append-only generation and episode evidence, with atomic live progress."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_progress(directory, data):
    staging = directory / "progress.tmp"
    staging.write_text(json.dumps({**data, "updated_at": timestamp()}, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    os.replace(staging, directory / "progress.json")


class RunLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = os.environ.get("MP3_RUN_ID", self.directory.name)
        planned = os.environ.get("MP3_PLANNED_TASKS")
        self.progress = {
            "run_id": self.run_id, "phase": "prediction", "status": "运行中",
            "planned": int(planned) if planned is not None else None,
            "completed": 0, "valid": 0, "failed": 0,
            "act_calls": 0, "charged_tools": 0, "completion_attempts": 0,
            "input_tokens": 0, "new_tokens": 0,
        }
        # A second runtime must never append into an existing prediction ledger.
        (self.directory / "generations.jsonl").touch(exist_ok=False)
        write_progress(self.directory, self.progress)

    def append(self, name, record, *, durable=False):
        entry = {"run_id": self.run_id, "timestamp": timestamp(), **record}
        with (self.directory / name).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            if durable:
                stream.flush()
                os.fsync(stream.fileno())

    def finish_task(self, record):
        self.append("trace.jsonl", record)
        totals = record["counts"]
        self.progress["completed"] += 1
        self.progress["valid"] += 1
        for key, value in totals.items():
            self.progress[key] = None if value is None or self.progress[key] is None else self.progress[key] + value
        self.append("metrics.jsonl", {"event": "task_completed", "task_id": record["task_id"],
                                      **totals, "public_pass": record["public_pass"]})
        write_progress(self.directory, self.progress)
