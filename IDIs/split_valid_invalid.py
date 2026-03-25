# import csv
# import logging
# import dataset_config
# from itertools import combinations
# import numpy as np
# import copy, os, ast
# import sys


# def bin_dataset_summary(dataset_summary):
#     binned_dataset_summary = copy.deepcopy(dataset_summary)
#     for attr, attr_info in binned_dataset_summary.items():
#         if attr_info.get("type") == "numerical":
#             start, end = map(int, attr_info["range"])
#             bin_edges = np.linspace(start, end, NUM_BINS + 1, dtype=int)
#             # Convert into categorical bins (e.g., "0~2", "3~5", ...)  
#             categories = [
#                 f"{bin_edges[i]}~{bin_edges[i+1]-1}" if i < len(bin_edges)-2 else f"{bin_edges[i]}~{bin_edges[i+1]}"
#                 for i in range(len(bin_edges) - 1)
#             ] 
#             # Update the dataset info
#             attr_info["type"] = "categorical"
#             attr_info["bins"] = categories   # add binning info
#             attr_info["range"] = [0, len(categories) - 1]  # new categorical index range
#     return binned_dataset_summary


# # 一つのデータインスタンスをbin化する関数（numerical変数のみ変える，欠損値(100)はそのままの値になる．）
# def binned_instance(instance, binned_dataset_summary):
#     binned_dataset_summary_list = [{"name": k, **v} for k, v in binned_dataset_summary.items()]
#     binned_dataset_summary_list = binned_dataset_summary_list[:-1]
#     binned_instance = instance.copy()
#     for i, info in enumerate(binned_dataset_summary_list):
#         if "bins" in info:
#             val = int(instance[i])
#             for idx, label in enumerate(info["bins"]):
#                 lo, hi = map(int, label.split("~"))
#                 if lo <= val <=hi:
#                     binned_instance[i] = idx
#                     break
#     return binned_instance


# # inpにはラベルを含まない
# def is_valid(inp, occur_table):
#     binned_inp = binned_instance(inp, binned_ds_summary)
#     pairs = list(combinations(range(len(binned_inp)), 2))
#     for idx, (i, j) in enumerate(pairs):
#         key = (binned_inp[i], binned_inp[j])
#         if key not in occur_table[idx] or occur_table[idx][key]==0:   # TO DO：値が0（定義域の範囲外）のときでも，bin化すると0になり，定義域内の値としてみなされてしまっている．
#             return False        
#     return True


# model_name = sys.argv[1]
# dataset_name = sys.argv[2]
# protected_name = sys.argv[3]
# method = sys.argv[4]
# runtime = sys.argv[5]

# NUM_BINS = 5

# ds_summary = getattr(dataset_config, dataset_name)
# binned_ds_summary = bin_dataset_summary(ds_summary)
        
# # occur_tableを読み込む
# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# occur_table_path = f"{root_dir}/datasets_prepared/occ_table/{dataset_name}_occ_table.csv"
# occur_table = []
# with open(occur_table_path, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         obj = ast.literal_eval(line)
#         occur_table.append(obj)


# # 検出された raw IDIs を読み込む
# with open(f'{root_dir}/IDIs/raw/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv', 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     raw_IDIs = [list(map(int, row)) for row in csvreader]


# pair_raw_IDIs = [
#     [raw_IDIs[i], raw_IDIs[i + 1]]
#     for i in range(0, len(raw_IDIs), 2)
# ]

# valid_IDIs = []
# invalid_IDIs = []

# for one_pair in pair_raw_IDIs:
#     IDI_1, IDI_2 = one_pair
#     # ラベルを除去
#     IDI_1_features = IDI_1[:-1]
#     IDI_2_features = IDI_2[:-1]    
#     if is_valid(IDI_1_features, occur_table) and is_valid(IDI_2_features, occur_table):
#         valid_IDIs.append(IDI_1)
#         valid_IDIs.append(IDI_2)
#     else:
#         invalid_IDIs.append(IDI_1)
#         invalid_IDIs.append(IDI_2)
        

# # csvファイルとして保存
# current_dir = os.path.dirname(os.path.abspath(__file__))
# valid_IDIs_path = f"{current_dir}/valid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv"
# invalid_IDIs_path = f"{current_dir}/invalid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv"
# logging.info(f"Saving the valid discriminatory instances to IDIs/valid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv")
# with open(valid_IDIs_path, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerows(valid_IDIs)
# logging.info(f"Saving the invalid discriminatory instances to IDIs/invalid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv")
# with open(invalid_IDIs_path, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerows(invalid_IDIs)
