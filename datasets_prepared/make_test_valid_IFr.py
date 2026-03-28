import os, sys, csv, ast, copy
import random
from itertools import product, combinations
import numpy as np
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from datasets_original import dataset_config


dataset_name = sys.argv[1]   # "Adult" or "Bank" or "Credit"
protected_attr = sys.argv[2]  # "age" or "race" or "sex"
NUM_BINS = 5

ds_summary = getattr(dataset_config, dataset_name)

protected_attr_index = list(ds_summary.keys()).index(protected_attr)
input_bounds = [v["range"] for k, v in ds_summary.items() if v["type"] != "output"]
attr_num = len(input_bounds)







# occur_tableを読み込む
current_dir = os.path.dirname(os.path.abspath(__file__))
occur_table_path = f"{current_dir}/occ_table/{dataset_name}_occ_table.csv"
occur_table = []
with open(occur_table_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        obj = ast.literal_eval(line)
        occur_table.append(obj)



def bin_dataset_summary(dataset_summary):
    binned_dataset_summary = copy.deepcopy(dataset_summary)
    for attr, attr_info in binned_dataset_summary.items():
        if attr_info.get("type") == "numerical":
            start, end = map(int, attr_info["range"])
            bin_edges = np.linspace(start, end, NUM_BINS + 1, dtype=int)

            # Convert into categorical bins (e.g., "0~2", "3~5", ...)  
            categories = [
                f"{bin_edges[i]}~{bin_edges[i+1]-1}" if i < len(bin_edges)-2 else f"{bin_edges[i]}~{bin_edges[i+1]}"
                for i in range(len(bin_edges) - 1)
            ] 
            # Update the dataset info
            attr_info["type"] = "categorical"
            attr_info["bins"] = categories   # add binning info
            attr_info["range"] = [0, len(categories) - 1]  # new categorical index range

    return binned_dataset_summary



# 一つのデータインスタンスをbin化する関数（numerical変数のみ変える，欠損値(100)はそのままの値になる．）
def binned_instance(instance, binned_dataset_summary):
    binned_dataset_summary_list = [{"name": k, **v} for k, v in binned_dataset_summary.items()]
    binned_dataset_summary_list = binned_dataset_summary_list[:-1]
    binned_instance = instance.copy()
    for i, info in enumerate(binned_dataset_summary_list):
        if "bins" in info:
            val = int(instance[i])
            for idx, label in enumerate(info["bins"]):
                lo, hi = map(int, label.split("~"))
                if lo <= val <=hi:
                    binned_instance[i] = idx
                    break   
    return binned_instance


# データが妥当であるかを判定，occur_table，binned_instance関数を用いる．
def is_valid(inp, occur_table):
    binned_inp = binned_instance(inp, binned_ds_summary)
    pairs = list(combinations(range(len(binned_inp)), 2))
    for idx, (i, j) in enumerate(pairs):
        key = (binned_inp[i], binned_inp[j])
        if key not in occur_table[idx] or occur_table[idx][key]==0:   # TO DO：値が0（定義域の範囲外）のときでも，bin化すると0になり，定義域内の値としてみなされてしまっている．
            return False        
    return True


def get_random_input_variants():
    random_input = [random.randint(min, max) for min, max in input_bounds]
    
    protected_min, protected_max = input_bounds[protected_attr_index]
    random_input_variants = [
        [val if i == protected_attr_index else random_input[i] for i in range(len(random_input))]
        for val in range(protected_min, protected_max + 1)
    ]
    
    return random_input_variants



def build_test_valid_IFr_set(samples,num_trials):
    test_valid_IFr_set=[]
    for _ in range(samples):
        for _ in range(num_trials):
            while True:
                variants = get_random_input_variants()
                valid_variants = [v for v in variants if is_valid(v, occur_table)]
                if len(valid_variants) >= 2:
                    break   
            test_valid_IFr_set.extend(valid_variants)
            test_valid_IFr_set.append([])
    return test_valid_IFr_set


binned_ds_summary = bin_dataset_summary(ds_summary)

samples = 100 #100
num_trials = 400 #400

test_valid_IFr_set = build_test_valid_IFr_set(samples,num_trials)  

cuurent_dir = os.path.dirname(os.path.abspath(__file__))
test_valid_IFr_path = f"{cuurent_dir}/test_valid_IFr/{dataset_name}_{protected_attr}_test_valid_IFr_set.csv"

with open(test_valid_IFr_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(test_valid_IFr_set)
    
