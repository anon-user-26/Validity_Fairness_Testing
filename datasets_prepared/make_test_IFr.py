import csv
import os
import random
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from datasets_original import dataset_config


dataset_name = sys.argv[1]      # "Adult", "Bank", or "Credit"
protected_attr = sys.argv[2]    # "age", "race", or "sex"

ds_summary = getattr(dataset_config, dataset_name)

protected_attr_index = list(ds_summary.keys()).index(protected_attr)
input_bounds = [v["range"] for k, v in ds_summary.items() if v["type"] != "output"]


def get_random_input_variants():
    """Generate one random input and enumerate all protected-attribute variants."""
    random_input = [random.randint(lower, upper) for lower, upper in input_bounds]

    protected_min, protected_max = input_bounds[protected_attr_index]
    random_input_variants = [
        [val if i == protected_attr_index else random_input[i] for i in range(len(random_input))]
        for val in range(protected_min, protected_max + 1)
    ]

    return random_input_variants


def build_test_IFr_set(samples, num_trials):
    """Build the test set used for IFr calculation."""
    test_IFr_set = []

    for _ in range(samples):
        for _ in range(num_trials):
            variants = get_random_input_variants()
            test_IFr_set.extend(variants)
            test_IFr_set.append([])  # Separator between groups

    return test_IFr_set


samples = 100
num_trials = 400

test_IFr_set = build_test_IFr_set(samples, num_trials)

current_dir = os.path.dirname(os.path.abspath(__file__))
test_IFr_path = f"{current_dir}/test_IFr/{dataset_name}_{protected_attr}_test_IFr_set.csv"

with open(test_IFr_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(test_IFr_set)