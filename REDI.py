import argparse
import subprocess
import sys
from pathlib import Path

RUNTIME = 3600

# Predefined experimental scenarios used in the paper.
SCENARIOS = [
    ("SVM", "Adult", "age"),
    ("SVM", "Credit", "age"),
    ("SVM", "Credit", "sex"),
    ("MLP", "Adult", "age"),
    ("MLP", "Adult", "race"),
    ("MLP", "Adult", "sex"),
    ("MLP", "Bank", "age"),
    ("MLP", "Credit", "age"),
    ("MLP", "Credit", "sex"),
    ("RanForest", "Bank", "age"),
    ("RanForest", "Credit", "age"),
    ("RanForest", "Credit", "sex"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the experimental pipeline for selected predefined scenarios."
    )
    parser.add_argument(
        "--model",
        required=True,
        help='Model name (e.g., SVM, MLP, RanForest) or "all".',
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help='Dataset name (e.g., Adult, Bank, Credit) or "all".',
    )
    parser.add_argument(
        "--protected",
        required=True,
        help='Protected attribute (e.g., age, race, sex) or "all".',
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated executions for each selected scenario.",
    )
    return parser.parse_args()


def select_scenarios(model, dataset, protected):
    selected = []

    for scenario_model, scenario_dataset, scenario_protected in SCENARIOS:
        if model != "all" and scenario_model != model:
            continue
        if dataset != "all" and scenario_dataset != dataset:
            continue
        if protected != "all" and scenario_protected != protected:
            continue
        selected.append((scenario_model, scenario_dataset, scenario_protected))

    if not selected:
        raise ValueError(
            f"No predefined scenario matches: "
            f"model={model}, dataset={dataset}, protected={protected}"
        )

    return selected


def run_command(command):
    display_command = [Path(command[1]).name] + command[2:]
    print(f"Running: {' '.join(display_command)}")
    subprocess.run(command, check=True)


def run_pipeline(root_dir, model_name, dataset_name, protected_name, runtime):
    
    # Split the original dataset into training and test sets.
    run_command(
        [
            sys.executable,
            str(root_dir / "datasets_prepared" / "split_train_test.py"),
            dataset_name,
        ]
    )

    # Construct occurrence tables used for validity checking (t-way interactions).
    run_command(
        [
            sys.executable,
            str(root_dir / "datasets_prepared" / "make_occ_table.py"),
            dataset_name,
        ]
    )

    # Generate test inputs for estimating IFr (fairness metric).
    run_command(
        [
            sys.executable,
            str(root_dir / "datasets_prepared" / "make_test_IFr.py"),
            dataset_name,
            protected_name,
        ]
    )

    # Generate valid test inputs for estimating valid-IFr.
    run_command(
        [
            sys.executable,
            str(root_dir / "datasets_prepared" / "make_test_valid_IFr.py"),
            dataset_name,
            protected_name,
        ]
    )

    # Train the original model.
    run_command(
        [
            sys.executable,
            str(root_dir / "models_trained" / "train.py"),
            model_name,
            dataset_name,
        ]
    )

    # Run fairness testing.
    run_command(
        [
            sys.executable,
            str(root_dir / "IFT" / "exp.py"),
            "--dataset_name",
            dataset_name,
            "--protected_attr",
            protected_name,
            "--model_name",
            model_name,
            "--method",
            "aft",
            "--runtime",
            str(runtime),
            "--repeat",
            "1",
        ]
    )

    # Retrain the model using the detected IDIs.
    run_command(
        [
            sys.executable,
            str(root_dir / "models_retrained" / "retrain.py"),
            model_name,
            dataset_name,
            protected_name,
            "aft",
            str(runtime),
        ]
    )


args = parse_args()
root_dir = Path(__file__).resolve().parent
scenarios = select_scenarios(args.model, args.dataset, args.protected)

for run_idx in range(args.runs):
    print(f"\n===== Run {run_idx + 1}/{args.runs} =====")
    for model_name, dataset_name, protected_name in scenarios:
        print(f"\n--- Scenario: model={model_name}, dataset={dataset_name}, protected={protected_name} ---")
        run_pipeline(
            root_dir=root_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            protected_name=protected_name,
            runtime=RUNTIME,
        )