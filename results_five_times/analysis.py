import os
import glob
import sys
import numpy as np
from scipy.stats import t


def partial_corr_with_p(x, y, z):
    """Compute partial correlation r_xy·z and its p-value."""
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]

    numerator = r_xy - r_xz * r_yz
    denominator = ((1 - r_xz**2) * (1 - r_yz**2)) ** 0.5

    if denominator == 0:
        return float("nan"), float("nan")

    r = numerator / denominator

    n = len(x)
    df = n - 3
    if df <= 0:
        return r, float("nan")

    t_value = r * np.sqrt(df / (1 - r**2))
    p_value = 2 * (1 - t.cdf(abs(t_value), df))

    return r, p_value


def multivariate_regression(x, y, z):
    """Compute z = alpha*x + beta*y + gamma and return coefficients and R^2."""
    X = np.column_stack([x, y, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    alpha, beta, gamma = coef
    z_pred = X @ coef

    ss_res = np.sum((z - z_pred)**2)
    ss_tot = np.sum((z - np.mean(z))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return alpha, beta, gamma, r2


def analysis(model_name, dataset_name, protected_name):
    base_dir = f"./{model_name}/{dataset_name}/{protected_name}"
    pattern = os.path.join(base_dir, f"{model_name}_{dataset_name}_{protected_name}_*_average.txt")
    all_files = glob.glob(pattern)

    valid_files = []
    for path in all_files:
        parts = os.path.basename(path).split("_")
        validity_str = parts[-2]
        try:
            float(validity_str)
            valid_files.append(path)
        except ValueError:
            pass

    valid_files = sorted(valid_files, key=lambda p: float(os.path.basename(p).split("_")[-2]))

    if len(valid_files) == 0:
        print("No valid average files found.")
        return

    validity_list = []
    L0_list = []
    L1_list = []
    bL0_list = []
    acc_imp_list = []
    ifr_imp_list = []
    valid_ifr_imp_list = []

    output_lines = []

    for path in valid_files:
        with open(path, "r") as f:
            lines = f.readlines()

        mid = {}
        for line in lines:
            if ":" not in line:
                continue
            key, vals = line.strip().split(":", 1)
            numbers = vals.strip().split()
            if len(numbers) < 3:
                continue
            mid[key] = float(numbers[1])

        v = mid["validity"]
        L0 = mid["average_pairwise_L0_distance"]
        L1 = mid["average_pairwise_L1_distance"]
        bL0 = mid["average_pairwise_binned_L0_distance"]

        acc_tr = mid["accuracy_trained"]
        acc_re = mid["accuracy_retrained"]
        ifr_tr = mid["IFr_trained"]
        ifr_re = mid["IFr_retrained"]
        v_ifr_tr = mid["valid_IFr_trained"]
        v_ifr_re = mid["valid_IFr_retrained"]

        acc_imp = 100 * (acc_re - acc_tr) / acc_tr
        ifr_imp = 100 * (ifr_tr - ifr_re) / ifr_tr
        valid_ifr_imp = 100 * (v_ifr_tr - v_ifr_re) / v_ifr_tr

        validity_list.append(v)
        L0_list.append(L0)
        L1_list.append(L1)
        bL0_list.append(bL0)
        acc_imp_list.append(acc_imp)
        ifr_imp_list.append(ifr_imp)
        valid_ifr_imp_list.append(valid_ifr_imp)

        output_lines.append(f"{v} {L0} {L1} {bL0} {acc_imp} {ifr_imp} {valid_ifr_imp}")

    v_arr = np.array(validity_list)
    L0_arr = np.array(L0_list)
    L1_arr = np.array(L1_list)
    bL0_arr = np.array(bL0_list)
    acc_arr = np.array(acc_imp_list)
    ifr_arr = np.array(ifr_imp_list)
    vifr_arr = np.array(valid_ifr_imp_list)

    # ----- PARTIAL CORR -----
    def pc(z_arr, control_arr):
        return partial_corr_with_p(v_arr, z_arr, control_arr)

    pc_results = {
        "acc_imp": {
            "L0": pc(acc_arr, L0_arr),
            "L1": pc(acc_arr, L1_arr),
            "bL0": pc(acc_arr, bL0_arr),
        },
        "IFr_imp": {
            "L0": pc(ifr_arr, L0_arr),
            "L1": pc(ifr_arr, L1_arr),
            "bL0": pc(ifr_arr, bL0_arr),
        },
        "valid_IFr_imp": {
            "L0": pc(vifr_arr, L0_arr),
            "L1": pc(vifr_arr, L1_arr),
            "bL0": pc(vifr_arr, bL0_arr),
        },
    }

    # ----- MULTIPLE REGRESSION -----
    def reg(y):
        
        return {
            "L0": multivariate_regression(v_arr, L0_arr, y),
            
            "L1": multivariate_regression(v_arr, L1_arr, y),
            "bL0": multivariate_regression(v_arr, bL0_arr, y),
        }

    reg_results = {
        "acc_imp": reg(acc_arr),
        "IFr_imp": reg(ifr_arr),
        "valid_IFr_imp": reg(vifr_arr),
    }

    # ----- OUTPUT -----
    output_path = os.path.join(base_dir, f"{model_name}_{dataset_name}_{protected_name}_analysis.txt")

    with open(output_path, "w") as out:

        out.write("=== Partial Correlations (with p-values) ===\n")
        for metric in pc_results:
            out.write(f"\n[{metric}]\n")
            for ctrl in ["L0", "L1", "bL0"]:
                r, p = pc_results[metric][ctrl]
                out.write(f"  r(validity, {metric} | {ctrl}) = {r}, p = {p}\n")

        out.write("\n\n=== Multiple Regression (z = alpha*validity + beta*distance + gamma) ===\n")
        for metric in reg_results:
            out.write(f"\n[{metric}]\n")
            for ctrl in ["L0", "L1", "bL0"]:
                alpha, beta, gamma, r2 = reg_results[metric][ctrl]
                out.write(f"  distance={ctrl}: alpha={alpha}, beta={beta}, gamma={gamma}, R2={r2}\n")

        out.write("\n\nvalidity L0 L1 binned_L0 acc_imp IFr_imp valid_IFr_imp\n")
        for line in output_lines:
            out.write(line + "\n")

    print(f"Saved analysis to: {output_path}")


if __name__ == "__main__":
    analysis(sys.argv[1], sys.argv[2], sys.argv[3])
