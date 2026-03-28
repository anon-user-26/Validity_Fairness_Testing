from pathlib import Path

import numpy as np


METRIC_NAMES = [
    "validity",
    "average_pairwise_L0_distance",
    "average_pairwise_L1_distance",
    "average_pairwise_binned_L0_distance",
    "accuracy_trained",
    "IFr_trained",
    "valid_IFr_trained",
    "accuracy_retrained",
    "IFr_retrained",
    "valid_IFr_retrained",
]


def compute_interval(values):
    """Return mean and mean ± 1.96 * std."""
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    lower = mean - 1.96 * std / np.sqrt(n)
    upper = mean + 1.96 * std / np.sqrt(n)
    return lower, mean, upper


def parse_result_file(file_path):
    """Read a result file and collect values for each metric."""
    metric_values = {name: [] for name in METRIC_NAMES}

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # Skip the header line.

    for line in lines:
        parts = line.split()
        if len(parts) != len(METRIC_NAMES):
            continue

        for metric_name, value in zip(METRIC_NAMES, parts):
            metric_values[metric_name].append(float(value))

    return metric_values


def write_average_file(output_path, metric_values):
    """Write summary statistics for each metric."""
    lines = []

    for metric_name in METRIC_NAMES:
        values = metric_values[metric_name]
        if not values:
            continue

        lower, mean, upper = compute_interval(values)
        lines.append(f"{metric_name}: {lower} {mean} {upper}")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def is_target_result_file(file_path):
    """
    Check whether the file name follows:
    <model>_<dataset>_<protected>_<validity>.txt
    """
    if file_path.suffix != ".txt":
        return False

    if file_path.stem.endswith("_average"):
        return False

    parts = file_path.stem.split("_")
    if len(parts) != 4:
        return False

    try:
        float(parts[3])
    except ValueError:
        return False

    return True


def build_expected_path(model_name, dataset_name, protected_name, validity):
    """Return the canonical path expected by the project structure."""
    return (
        Path(model_name)
        / dataset_name
        / protected_name
        / f"{model_name}_{dataset_name}_{protected_name}_{validity}.txt"
    )


def process_file(file_path):
    """Generate an average file for one result file."""
    parts = file_path.stem.split("_")
    model_name, dataset_name, protected_name, validity = parts

    expected_path = build_expected_path(
        model_name, dataset_name, protected_name, validity
    )

    # Skip files that do not match the expected directory structure.
    if file_path != expected_path:
        return

    metric_values = parse_result_file(file_path)

    output_path = file_path.with_name(
        f"{model_name}_{dataset_name}_{protected_name}_{validity}_average.txt"
    )
    write_average_file(output_path, metric_values)

    print(f"Created: {output_path}")


def main():
    """Search result files recursively and generate average files."""
    for file_path in Path(".").rglob("*.txt"):
        if is_target_result_file(file_path):
            process_file(file_path)


if __name__ == "__main__":
    main()