from pathlib import Path

import numpy as np
from scipy.stats import t


def partial_corr_with_p(x, y, z):
    """Return the partial correlation r(x, y | z) and its p-value."""
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]

    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    if denominator == 0:
        return float("nan"), float("nan")

    r_value = numerator / denominator

    n = len(x)
    df = n - 3
    if df <= 0:
        return r_value, float("nan")

    t_value = r_value * np.sqrt(df / (1 - r_value**2))
    p_value = 2 * (1 - t.cdf(abs(t_value), df))
    return r_value, p_value


def multivariate_regression(validity, distance, target):
    """Fit target = alpha * validity + beta * distance + gamma."""
    design_matrix = np.column_stack([validity, distance, np.ones(len(validity))])
    coef, _, _, _ = np.linalg.lstsq(design_matrix, target, rcond=None)
    alpha, beta, gamma = coef

    predicted = design_matrix @ coef
    ss_res = np.sum((target - predicted) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return alpha, beta, gamma, r_squared


def simple_corr_with_p(x, y):
    """Return the Pearson correlation coefficient and its p-value."""
    r_value = np.corrcoef(x, y)[0, 1]

    n = len(x)
    df = n - 2
    if df <= 0:
        return r_value, float("nan")

    t_value = r_value * np.sqrt(df / (1 - r_value**2))
    p_value = 2 * (1 - t.cdf(abs(t_value), df))
    return r_value, p_value


def parse_average_file(file_path):
    """Extract the mean value of each metric from an *_average.txt file."""
    metrics = {}

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            key, values = line.strip().split(":", 1)
            numbers = values.strip().split()

            if len(numbers) < 3:
                continue

            metrics[key] = float(numbers[1])

    return metrics


def collect_average_files(base_dir, model_name, dataset_name, protected_name):
    """Collect valid average-result files sorted by validity."""
    pattern = f"{model_name}_{dataset_name}_{protected_name}_*_average.txt"
    average_files = []

    for file_path in base_dir.glob(pattern):
        parts = file_path.stem.split("_")

        if len(parts) < 5:
            continue

        validity_str = parts[-2]

        try:
            float(validity_str)
        except ValueError:
            continue

        average_files.append(file_path)

    average_files.sort(key=lambda path: float(path.stem.split("_")[-2]))
    return average_files


def analyze_directory(model_name, dataset_name, protected_name):
    """Generate one analysis file for a single model/dataset/protected combination."""
    base_dir = Path(model_name) / dataset_name / protected_name

    average_files = collect_average_files(
        base_dir, model_name, dataset_name, protected_name
    )

    if not average_files:
        print(f"No valid average files found in: {base_dir}")
        return

    validity_values = []
    l0_values = []
    l1_values = []
    bl0_values = []
    acc_improvement_values = []
    ifr_improvement_values = []
    valid_ifr_improvement_values = []
    output_rows = []

    for file_path in average_files:
        metrics = parse_average_file(file_path)

        validity = metrics["validity"]
        l0 = metrics["average_pairwise_L0_distance"]
        l1 = metrics["average_pairwise_L1_distance"]
        bl0 = metrics["average_pairwise_binned_L0_distance"]

        acc_trained = metrics["accuracy_trained"]
        ifr_trained = metrics["IFr_trained"]
        valid_ifr_trained = metrics["valid_IFr_trained"]

        acc_retrained = metrics["accuracy_retrained"]
        ifr_retrained = metrics["IFr_retrained"]
        valid_ifr_retrained = metrics["valid_IFr_retrained"]

        acc_improvement = 100 * (acc_retrained - acc_trained) / acc_trained
        ifr_improvement = 100 * (ifr_trained - ifr_retrained) / ifr_trained
        valid_ifr_improvement = (
            100 * (valid_ifr_trained - valid_ifr_retrained) / valid_ifr_trained
        )

        validity_values.append(validity)
        l0_values.append(l0)
        l1_values.append(l1)
        bl0_values.append(bl0)
        acc_improvement_values.append(acc_improvement)
        ifr_improvement_values.append(ifr_improvement)
        valid_ifr_improvement_values.append(valid_ifr_improvement)

        output_rows.append(
            f"{validity} {l0} {l1} {bl0} "
            f"{acc_trained} {ifr_trained} {valid_ifr_trained} "
            f"{acc_retrained} {ifr_retrained} {valid_ifr_retrained} "
            f"{acc_improvement} {ifr_improvement} {valid_ifr_improvement}"
        )

    validity_arr = np.array(validity_values)
    l0_arr = np.array(l0_values)
    l1_arr = np.array(l1_values)
    bl0_arr = np.array(bl0_values)
    acc_arr = np.array(acc_improvement_values)
    ifr_arr = np.array(ifr_improvement_values)
    valid_ifr_arr = np.array(valid_ifr_improvement_values)

    corr_l0, p_l0 = simple_corr_with_p(validity_arr, l0_arr)
    corr_l1, p_l1 = simple_corr_with_p(validity_arr, l1_arr)
    corr_bl0, p_bl0 = simple_corr_with_p(validity_arr, bl0_arr)

    partial_corr_results = {
        "acc_imp": {
            "L0": partial_corr_with_p(validity_arr, acc_arr, l0_arr),
            "L1": partial_corr_with_p(validity_arr, acc_arr, l1_arr),
            "bL0": partial_corr_with_p(validity_arr, acc_arr, bl0_arr),
        },
        "IFr_imp": {
            "L0": partial_corr_with_p(validity_arr, ifr_arr, l0_arr),
            "L1": partial_corr_with_p(validity_arr, ifr_arr, l1_arr),
            "bL0": partial_corr_with_p(validity_arr, ifr_arr, bl0_arr),
        },
        "valid_IFr_imp": {
            "L0": partial_corr_with_p(validity_arr, valid_ifr_arr, l0_arr),
            "L1": partial_corr_with_p(validity_arr, valid_ifr_arr, l1_arr),
            "bL0": partial_corr_with_p(validity_arr, valid_ifr_arr, bl0_arr),
        },
    }

    regression_results = {
        "acc_imp": {
            "L0": multivariate_regression(validity_arr, l0_arr, acc_arr),
            "L1": multivariate_regression(validity_arr, l1_arr, acc_arr),
            "bL0": multivariate_regression(validity_arr, bl0_arr, acc_arr),
        },
        "IFr_imp": {
            "L0": multivariate_regression(validity_arr, l0_arr, ifr_arr),
            "L1": multivariate_regression(validity_arr, l1_arr, ifr_arr),
            "bL0": multivariate_regression(validity_arr, bl0_arr, ifr_arr),
        },
        "valid_IFr_imp": {
            "L0": multivariate_regression(validity_arr, l0_arr, valid_ifr_arr),
            "L1": multivariate_regression(validity_arr, l1_arr, valid_ifr_arr),
            "bL0": multivariate_regression(validity_arr, bl0_arr, valid_ifr_arr),
        },
    }

    output_path = base_dir / f"{model_name}_{dataset_name}_{protected_name}_analysis.txt"

    with output_path.open("w", encoding="utf-8") as out:
        out.write("=== Correlation (validity vs distances) ===\n")
        out.write(
            f"r(validity, L0) = {corr_l0}, p = {p_l0}\n"
            f"r(validity, L1) = {corr_l1}, p = {p_l1}\n"
            f"r(validity, bL0) = {corr_bl0}, p = {p_bl0}\n\n"
        )

        out.write("=== Partial Correlations (with p-values) ===\n")
        for metric_name, controls in partial_corr_results.items():
            out.write(f"\n[{metric_name}]\n")
            for control_name in ["L0", "L1", "bL0"]:
                r_value, p_value = controls[control_name]
                out.write(
                    f"  r(validity, {metric_name} | {control_name}) = "
                    f"{r_value}, p = {p_value}\n"
                )

        out.write(
            "\n\n=== Multiple Regression "
            "(target = alpha * validity + beta * distance + gamma) ===\n"
        )
        for metric_name, controls in regression_results.items():
            out.write(f"\n[{metric_name}]\n")
            for control_name in ["L0", "L1", "bL0"]:
                alpha, beta, gamma, r_squared = controls[control_name]
                out.write(
                    f"  distance={control_name}: "
                    f"alpha={alpha}, beta={beta}, gamma={gamma}, R2={r_squared}\n"
                )

        out.write(
            "\n\nvalidity L0 L1 binned_L0 "
            "acc_trained IFr_trained valid_IFr_trained "
            "acc_retrained IFr_retrained valid_IFr_retrained "
            "acc_imp IFr_imp valid_IFr_imp\n"
        )
        for row in output_rows:
            out.write(row + "\n")

    print(f"Saved analysis to: {output_path}")


def run_all_analyses():
    """Run analysis for every ./model/dataset/protected directory."""
    root_dir = Path(".")

    for model_dir in root_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            for protected_dir in dataset_dir.iterdir():
                if not protected_dir.is_dir():
                    continue

                model_name = model_dir.name
                dataset_name = dataset_dir.name
                protected_name = protected_dir.name

                print(
                    f"Running analysis for "
                    f"{model_name}/{dataset_name}/{protected_name}"
                )

                try:
                    analyze_directory(model_name, dataset_name, protected_name)
                except Exception as exc:
                    print(
                        f"Error in {model_name}/{dataset_name}/{protected_name}: {exc}"
                    )


if __name__ == "__main__":
    run_all_analyses()