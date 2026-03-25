# occurrence tableをつくるコード

import sys, copy, os
import numpy as np
from itertools import product, combinations
import dataset_config # dataset_config (二つある)
from itertools import combinations
import pandas as pd


def init_occurrence_table(binned_dataset_summary, STRENGTH_T):
    occurrence_table = []
    assert(STRENGTH_T==1 or STRENGTH_T==2)

    binned_dataset_summary_list = [{"name": k, **v} for k, v in binned_dataset_summary.items()]
    binned_dataset_summary_list = binned_dataset_summary_list[:-1]

    if (STRENGTH_T == 1):
        for idx, attr_info in enumerate(binned_dataset_summary_list):
            print(f"Index={idx},  Value={attr_info}")
            rng = attr_info['range']
            attr_comb_dict = {}
            for attr_val in range(rng[0], rng[1] + 1):
                attr_comb_dict.update({attr_val: 0})
            occurrence_table.append(attr_comb_dict)
    elif (STRENGTH_T == 2):
        #print("binned_dataset_summary_list", binned_dataset_summary_list)
        for attr_i, attr_j in combinations(range(NUM_ATTR), 2):
            #print("2-way feature combi:", attr_i, attr_j)
            rng_i_lo, rng_i_hi = binned_dataset_summary_list[attr_i]['range']
            rng_j_lo, rng_j_hi = binned_dataset_summary_list[attr_j]['range']

            # inclusive range: [lo, hi] → range(lo, hi+1)
            rng_i = range(rng_i_lo, rng_i_hi + 1)
            rng_j = range(rng_j_lo, rng_j_hi + 1)

            # 全組合せを (idx_i, idx_j) → 0 で初期化
            attr_comb_dict = {(i, j): 0 for i, j in product(rng_i, rng_j)}
            occurrence_table.append(attr_comb_dict)
            #print(attr_comb_dict)
                
    return occurrence_table



def update_occurrence_table(occur_table, df_td, binned_dataset_summary):
    assert(STRENGTH_T==1 or STRENGTH_T==2)
    
    if STRENGTH_T == 1:
        for row in df_td.itertuples(index=False, name=None): 
            row = list(row[:-1])         
            binned_row = binned_instance(row, binned_dataset_summary)
            for idx, key in enumerate(binned_row):
                # keyが欠損値でない場合のみ，occur_tableを更新する
                if key in occur_table[idx]:                    
                    occur_table[idx][key] += 1
    
    if STRENGTH_T == 2:
        pairs = list(combinations(range(NUM_ATTR), 2)) 
        for row in df_td.itertuples(index=False, name=None): 
            row = list(row[:-1])           
            binned_row = binned_instance(row, binned_dataset_summary)
            for idx, (i, j) in enumerate(pairs):
                key = (binned_row[i], binned_row[j])
                if key in occur_table[idx]:                    
                    occur_table[idx][key] += 1
        
    return occur_table   



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


dataset_name = sys.argv[1]   # "Adult" or "Bank" or "Credit"
NUM_BINS = 5
STRENGTH_T = 2

ds_summary = getattr(dataset_config, dataset_name)

NUM_ATTR = len(ds_summary)-1


# Load training dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = f"{current_dir}/train/{dataset_name}_train.csv"

df_td = pd.read_csv(train_path) 

binned_ds_summary = bin_dataset_summary(ds_summary)

occur_table = init_occurrence_table(binned_ds_summary, STRENGTH_T)
occur_table = update_occurrence_table(occur_table, df_td, binned_ds_summary)

occur_table_path = f"{current_dir}/occ_table/{dataset_name}_occ_table.csv"

with open(occur_table_path, "w", encoding="utf-8") as f:
    for row in occur_table:
        f.write(str(row) + "\n")