import ast
import copy
import csv
import os
import random
import sys
from itertools import combinations

import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from datasets_original import dataset_config


dataset_name = sys.argv[1]      # "Adult", "Bank", or "Credit"
protected_attr = sys.argv[2]    # "age", "race", or "sex"
NUM_BINS = 5

ds_summary = getattr(dataset_config, dataset_name)

protected_attr_index = list(ds_summary.keys()).index(protected_attr)
input_bounds = [v["range"] for k, v in ds_summary.items() if v["type"] != "output"]


# Load the occurrence table inferred from the training data.
current_dir = os.path.dirname(os.path.abspath(__file__))
occur_table_path = f"{current_dir}/occ_table/{dataset_name}_occ_table.csv"

occur_table = []
with open(occur_table_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        obj = ast.literal_eval(line)
        occur_table.append(obj)


def bin_dataset_summary(dataset_summary):
    """Convert numerical attributes into categorical bins."""
    binned_dataset_summary = copy.deepcopy(dataset_summary)

    for attr, attr_info in binned_dataset_summary.items():
        if attr_info.get("type") == "numerical":
            start, end = map(int, attr_info["range"])
            bin_edges = np.linspace(start, end, NUM_BINS + 1, dtype=int)

            categories = [
                f"{bin_edges[i]}~{bin_edges[i + 1] - 1}"
                if i < len(bin_edges) - 2
                else f"{bin_edges[i]}~{bin_edges[i + 1]}"
                for i in range(len(bin_edges) - 1)
            ]

            attr_info["type"] = "categorical"
            attr_info["bins"] = categories
            attr_info["range"] = [0, len(categories) - 1]

    return binned_dataset_summary


def binned_instance(instance, binned_dataset_summary):
    """Map numerical values in an instance to their corresponding bin indices."""
    binned_dataset_summary_list = [{"name": k, **v} for k, v in binned_dataset_summary.items()]
    binned_dataset_summary_list = binned_dataset_summary_list[:-1]  # Exclude output attribute

    converted_instance = instance.copy()

    for i, info in enumerate(binned_dataset_summary_list):
        if "bins" in info:
            val = int(instance[i])
            for idx, label in enumerate(info["bins"]):
                lower, upper = map(int, label.split("~"))
                if lower <= val <= upper:
                    converted_instance[i] = idx
                    break

    return converted_instance


def is_valid(inp, occur_table):
    """Check whether an input instance satisfies all observed pairwise co-occurrences."""
    binned_inp = binned_instance(inp, binned_ds_summary)
    pairs = list(combinations(range(len(binned_inp)), 2))

    for idx, (i, j) in enumerate(pairs):
        key = (binned_inp[i], binned_inp[j])
        if key not in occur_table[idx] or occur_table[idx][key] == 0:
            return False

    return True


def get_random_input_variants():
    """Generate one random input and enumerate all protected-attribute variants."""
    random_input = [random.randint(lower, upper) for lower, upper in input_bounds]

    protected_min, protected_max = input_bounds[protected_attr_index]
    random_input_variants = [
        [val if i == protected_attr_index else random_input[i] for i in range(len(random_input))]
        for val in range(protected_min, protected_max + 1)
    ]

    return random_input_variants


def build_test_valid_IFr_set(samples, num_trials):
    """Build the test set used for valid-IFr calculation."""
    test_valid_IFr_set = []

    for _ in range(samples):
        for _ in range(num_trials):
            while True:
                variants = get_random_input_variants()
                valid_variants = [v for v in variants if is_valid(v, occur_table)]
                if len(valid_variants) >= 2:
                    break

            test_valid_IFr_set.extend(valid_variants)
            test_valid_IFr_set.append([])  # Separator between groups

    return test_valid_IFr_set


binned_ds_summary = bin_dataset_summary(ds_summary)

samples = 100
num_trials = 400

test_valid_IFr_set = build_test_valid_IFr_set(samples, num_trials)

current_dir = os.path.dirname(os.path.abspath(__file__))
test_valid_IFr_path = (
    f"{current_dir}/test_valid_IFr/{dataset_name}_{protected_attr}_test_valid_IFr_set.csv"
)

with open(test_valid_IFr_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(test_valid_IFr_set)