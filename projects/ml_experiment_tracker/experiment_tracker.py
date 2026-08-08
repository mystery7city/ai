from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def parse_key_value(items: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")

        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if not key:
            raise ValueError(f"Empty key in item: {item}")

        values[key] = coerce_value(raw_value)

    return values


def coerce_value(value: str) -> Any:
    lowered = value.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "experiment"


def record_experiment(
    name: str,
    model: str,
    dataset: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    notes: str = "",
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> Path:
    experiments_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = unique_experiment_id(slugify(name), experiments_dir)
    experiment = {
        "id": experiment_id,
        "name": name,
        "model": model,
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "parameters": parameters,
        "metrics": metrics,
        "notes": notes,
    }

    output_path = experiments_dir / f"{experiment_id}.json"
    output_path.write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def unique_experiment_id(base_id: str, experiments_dir: Path) -> str:
    candidate = base_id
    counter = 2

    while (experiments_dir / f"{candidate}.json").exists():
        candidate = f"{base_id}_{counter}"
        counter += 1

    return candidate


def load_experiments(experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR) -> list[dict[str, Any]]:
    if not experiments_dir.exists():
        return []

    experiments = []
    for path in sorted(experiments_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            experiment = json.load(file)
        experiment["_file"] = path.name
        experiments.append(experiment)

    return experiments


def compare_experiments(experiments: list[dict[str, Any]]) -> str:
    if not experiments:
        return "No experiments found."

    metric_names = sorted(
        {
            metric_name
            for experiment in experiments
            for metric_name in experiment.get("metrics", {}).keys()
        }
    )
    columns = ["id", "model", "dataset", *metric_names, "notes"]
    rows = []

    for experiment in experiments:
        metrics = experiment.get("metrics", {})
        rows.append(
            [
                str(experiment.get("id", "")),
                str(experiment.get("model", "")),
                str(experiment.get("dataset", "")),
                *[format_cell(metrics.get(metric_name, "")) for metric_name in metric_names],
                str(experiment.get("notes", "")),
            ]
        )

    return format_table(columns, rows)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_table(columns: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(column), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    divider = "-+-".join("-" * width for width in widths)
    body = [
        " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    ]

    return "\n".join([header, divider, *body])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record and compare ML experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    log_parser = subparsers.add_parser("log", help="Save a new experiment JSON file.")
    log_parser.add_argument("--name", required=True, help="Experiment name.")
    log_parser.add_argument("--model", required=True, help="Model name.")
    log_parser.add_argument("--dataset", required=True, help="Dataset name.")
    log_parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Parameter entry. Can be used multiple times.",
    )
    log_parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Metric entry. Can be used multiple times.",
    )
    log_parser.add_argument("--notes", default="", help="Short experiment notes.")

    subparsers.add_parser("compare", help="Print a comparison table.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "log":
        try:
            output_path = record_experiment(
                name=args.name,
                model=args.model,
                dataset=args.dataset,
                parameters=parse_key_value(args.param),
                metrics=parse_key_value(args.metric),
                notes=args.notes,
            )
        except ValueError as error:
            parser.error(str(error))

        print(f"Saved experiment: {output_path}")
        return

    if args.command == "compare":
        print(compare_experiments(load_experiments()))


if __name__ == "__main__":
    main()

