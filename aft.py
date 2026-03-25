import random
import time
from sklearn.tree import DecisionTreeClassifier
import csv
from utils.PathSearcher import PathSearcher, IntervalPool
import logging
import itertools
from datasets_original import dataset_config
from itertools import combinations
import numpy as np
import copy, os, ast
import pandas as pd

def load_train_size(dataset_name):
    filename = f"datasets_prepared/train/{dataset_name}_train.csv"
    with open(filename, "r") as f:
        return sum(1 for _ in f) - 1

class AFT:
    def __init__(self, black_box_model, protected_list, no_train_data_sample, show_logging=False):
        self.black_box_model = black_box_model
        self.train_data = list()
        self.disc_data = list()
        self.test_data = list()
        self.protected_list = [self.black_box_model.feature_list[i] for i in protected_list] # ["age"]
        self.protected_list_no = protected_list # [0] (age index)
        self.no_train_data_sample = no_train_data_sample
        self.no_test = 0
        self.no_disc = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        self.protected_value_comb = self.generate_protected_value_combination() # [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
        self.no_prot = len(protected_list)
        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def generate_protected_value_combination(self):
        res = list()
        for index_protected in self.protected_list_no:
            MinMax = self.black_box_model.data_range[index_protected] # [1, 9]
            res.append(list(range(MinMax[0], MinMax[1] + 1)))
        return list(itertools.product(*res))

    def create_train_data(self, num):
        self.train_data = list()
        black_model = self.black_box_model
        data_range = black_model.data_range
        for _ in range(num):
            temp = list()
            for i in range(black_model.no_attr):
                temp.append(random.randint(data_range[i][0], data_range[i][1]))
            temp.append(int(black_model.predict([temp])))
            self.train_data.append(temp)

    def train_approximate_DT(self, max_leaf_nodes):
        X = [item[:-1] for item in self.train_data]
        Y = [item[-1] for item in self.train_data]
        clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                       random_state=None, max_leaf_nodes=max_leaf_nodes)
        return clf.fit(X, Y)

    def test(self, runtime=None, max_leaf_nodes=1000, max_test_data=None, label=("res",0),
             dt_search_mode="random+flip", check_type="themis", MaxTry=10000, MaxDiscPathPair=100, max_train_data_each_path=10,
             max_sample_each_path=100):
        start_real_time = time.time()
        start_cpu_time = time.process_time()
        restart_flag = True
        self.no_test = 0
        no_new_train_count = 0
        loop = 0
        IntervalP = IntervalPool()

        # Main loop of AFT (Algorithm 2 in paper)
        logging.info(f"Starting fairness test -- {label[0]}")
        
        
        # 以降，valid_IDIs と invalid_IDIsを取得するために追加したコード
        
        method, model_name, dataset_name, protected_attr, runtime_str = label[0].split("-")
        NUM_BINS = 5
       
        
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
        
        # inpにはラベルを含まない
        def is_valid(inp, occur_table):
            binned_inp = binned_instance(inp, binned_ds_summary)
            pairs = list(combinations(range(len(binned_inp)), 2))
            for idx, (i, j) in enumerate(pairs):
                key = (binned_inp[i], binned_inp[j])
                if key not in occur_table[idx] or occur_table[idx][key]==0:   # TO DO：値が0（定義域の範囲外）のときでも，bin化すると0になり，定義域内の値としてみなされてしまっている．
                    return False        
            return True
        
        ds_summary = getattr(dataset_config, dataset_name)
        binned_ds_summary = bin_dataset_summary(ds_summary)

        # occur_tableを読み込む
        root_dir = os.path.dirname(os.path.abspath(__file__))
        occur_table_path = f"{root_dir}/datasets_prepared/occ_table/{dataset_name}_occ_table.csv"
        occur_table = []
        with open(occur_table_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                obj = ast.literal_eval(line)
                occur_table.append(obj)

        valid_IDIs = []
        invalid_IDIs = []
        
        while True:
            loop += 1
            if (runtime is not None) and (time.process_time() - start_cpu_time >= runtime):
                break
            if (max_test_data is not None) and (self.no_test >= max_test_data):
                break
            
            retrain_dataset_size = int(load_train_size(dataset_name)*0.1)
            
            if len(valid_IDIs) >= retrain_dataset_size and len(invalid_IDIs) >= retrain_dataset_size:
                break

            # Generate training data from input space of CuT at random.
            if restart_flag:
                self.create_train_data(self.no_train_data_sample)
                restart_flag = False

            # (Re-)train an approximate decision tree model with training data, as an approximation of CuT.
            DT = self.train_approximate_DT(max_leaf_nodes=max_leaf_nodes)

            # Generate test cases and then identify discriminatory instances through path samping and random search
            # (see the function sampler.sample()).
            sampler = PathSearcher(DT=DT, CuT=self.black_box_model, data_range=self.black_box_model.data_range,
                                   protected_value_comb=self.protected_value_comb, protected_list_no=self.protected_list_no, IntervalP=IntervalP)
            satFlag = sampler.sample(dt_search_mode=dt_search_mode, check_type=check_type, MaxTry=MaxTry, MaxDiscPathPair=MaxDiscPathPair,
                                     max_train_data_each_path=max_train_data_each_path, max_sample_each_path=max_sample_each_path)

            if satFlag:
                # If at least one test case is found,
                # update test cases and discriminatory instances
                test_data = sampler.get_test_data()
                self.no_test += len(test_data) // 2
                self.test_data += test_data
                new_disc_data = sampler.get_disc_data()
                self.no_disc += len(new_disc_data) // 2
                if len(new_disc_data) == 0:
                    # If no one discriminatory instance is found in this iteration, then restart
                    restart_flag = True
                    logging.info(f"Restarting due to not finding any discriminatory data in this loop")
                    logging.info(f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}")
                    continue
                else:
                    self.disc_data += new_disc_data
                    
                    
                    # new_disc_dataを valid と invalid に分けて，それぞれ，valid_IDIs と invalid_IDIs に追加
                    pair_raw_IDIs = [
                        [new_disc_data[i], new_disc_data[i + 1]]
                        for i in range(0, len(new_disc_data), 2)
                    ]
                    
                    for one_pair in pair_raw_IDIs:
                        IDI_1, IDI_2 = one_pair
                        # ラベルを除去
                        IDI_1_features = IDI_1[:-1]
                        IDI_2_features = IDI_2[:-1]    
                        if is_valid(IDI_1_features, occur_table) and is_valid(IDI_2_features, occur_table):
                            valid_IDIs.append(IDI_1)
                            valid_IDIs.append(IDI_2)
                        else:
                            invalid_IDIs.append(IDI_1)
                            invalid_IDIs.append(IDI_2)
                    

                # Update the training data
                new_train_data = sampler.get_train_data()
                self.train_data += new_train_data
                if len(new_train_data) == 0:
                    no_new_train_count += 1
                    if no_new_train_count >= 5:
                        restart_flag = True
                        no_new_train_count = 0
                else:
                    no_new_train_count = 0
            else:
                # If no one test case could be found from the decision tree, then restart the loop
                restart_flag = True
                logging.info(f"Restarting due to not finding any test cases in this loop")
            logging.info(f"Loop {loop}: #Disc={self.no_disc}, #Test={self.no_test}")

        self.real_time_consumed = time.time() - start_real_time
        self.cpu_time_consumed = time.process_time() - start_cpu_time
        # Save the results of identified discriminatory instances and generated test cases
        logging.info(f"The fairness test is completed")
        # logging.info(f"Saving the generated test cases to TestData/{label[0]}-{label[1]}.csv")
        # with open(f'TestData/{label[0]}-{label[1]}.csv', 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerows(self.test_data)
        # logging.info(f"Saving the detected discriminatory instances to DiscData/raw/{label[0]}-{label[1]}.csv")
        root_dir = os.path.dirname(os.path.abspath(__file__))
        raw_IDIs_path = f"{root_dir}/IDIs/raw/{label[0]}.csv"
        logging.info(f"Saving the detected discriminatory instances to IDIs/raw/{label[0]}.csv")
        with open(raw_IDIs_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.disc_data)
        
        valid_IDIs_path = f"{root_dir}/IDIs/valid/{label[0]}.csv"
        logging.info(f"Saving the valid discriminatory instances to IDIs/valid/{label[0]}.csv")
        with open(valid_IDIs_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(valid_IDIs)
            
        invalid_IDIs_path = f"{root_dir}/IDIs/invalid/{label[0]}.csv"
        logging.info(f"Saving the invalid discriminatory instances to IDIs/invalid/{label[0]}.csv")
        with open(invalid_IDIs_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(invalid_IDIs)
            
            