import sys
import os
import copy
from itertools import product, combinations

import numpy as np
import pandas as pd

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from datasets_original import dataset_config


def init_occurrence_table(binned_dataset_summary, strength_t):
    """Initialize occurrence table for t-way feature combinations."""
    assert strength_t in [1, 2]

    occurrence_table = []
    summary_list = [{"name": k, **v} for k, v in binned_dataset_summary.items()][:-1]

    if strength_t == 1:
        for attr_info in summary_list:
            rng = attr_info["range"]
            occurrence_table.append({val: 0 for val in range(rng[0], rng[1] + 1)})

    elif strength_t == 2:
        for i, j in combinations(range(NUM_ATTR), 2):
            rng_i = range(summary_list[i]["range"][0], summary_list[i]["range"][1] + 1)
            rng_j = range(summary_list[j]["range"][0], summary_list[j]["range"][1] + 1)
            occurrence_table.append({(a, b): 0 for a, b in product(rng_i, rng_j)})

    return occurrence_table


def update_occurrence_table(occ_table, df, binned_dataset_summary):
    """Update occurrence table using the dataset."""
    assert STRENGTH_T in [1, 2]

    if STRENGTH_T == 1:
        for row in df.itertuples(index=False, name=None):
            binned_row = binned_instance(list(row[:-1]), binned_dataset_summary)
            for idx, key in enumerate(binned_row):
                if key in occ_table[idx]:
                    occ_table[idx][key] += 1

    elif STRENGTH_T == 2:
        pairs = list(combinations(range(NUM_ATTR), 2))
        for row in df.itertuples(index=False, name=None):
            binned_row = binned_instance(list(row[:-1]), binned_dataset_summary)
            for idx, (i, j) in enumerate(pairs):
                key = (binned_row[i], binned_row[j])
                if key in occ_table[idx]:
                    occ_table[idx][key] += 1

    return occ_table


def binned_instance(instance, binned_dataset_summary):
    """Convert a single instance into binned representation."""
    summary_list = [{"name": k, **v} for k, v in binned_dataset_summary.items()][:-1]
    binned = instance.copy()

    for i, info in enumerate(summary_list):
        if "bins" in info:
            val = int(instance[i])
            for idx, label in enumerate(info["bins"]):
                lo, hi = map(int, label.split("~"))
                if lo <= val <= hi:
                    binned[i] = idx
                    break

    return binned


def bin_dataset_summary(dataset_summary):
    """Discretize numerical attributes into categorical bins."""
    binned_summary = copy.deepcopy(dataset_summary)

    for attr, info in binned_summary.items():
        if info.get("type") == "numerical":
            start, end = map(int, info["range"])
            edges = np.linspace(start, end, NUM_BINS + 1, dtype=int)

            bins = [
                f"{edges[i]}~{edges[i+1]-1}" if i < len(edges) - 2 else f"{edges[i]}~{edges[i+1]}"
                for i in range(len(edges) - 1)
            ]

            info["type"] = "categorical"
            info["bins"] = bins
            info["range"] = [0, len(bins) - 1]

    return binned_summary


# Configuration
dataset_name = sys.argv[1]  # e.g., "Adult", "Bank", "Credit"
NUM_BINS = 5
STRENGTH_T = 2

ds_summary = getattr(dataset_config, dataset_name)
NUM_ATTR = len(ds_summary) - 1


# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, "train", f"{dataset_name}_train.csv")

df = pd.read_csv(train_path)


# Build occurrence table
binned_summary = bin_dataset_summary(ds_summary)

occ_table = init_occurrence_table(binned_summary, STRENGTH_T)
occ_table = update_occurrence_table(occ_table, df, binned_summary)


# Save result
output_path = os.path.join(current_dir, "occ_table", f"{dataset_name}_occ_table.csv")

with open(output_path, "w", encoding="utf-8") as f:
    for row in occ_table:
        f.write(str(row) + "\n")