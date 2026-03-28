import ast
import copy
import csv
import itertools
import logging
import os
import random
import sys
import time
from itertools import combinations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from utils.PathSearcher import PathSearcher, IntervalPool

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from datasets_original import dataset_config


def load_train_size(dataset_name):
    """Return the number of training instances excluding the header row."""
    filename = f"{root_dir}/datasets_prepared/train/{dataset_name}_train.csv"
    with open(filename, "r") as f:
        return sum(1 for _ in f) - 1


class AFT:
    def __init__(self, black_box_model, protected_list, no_train_data_sample, show_logging=False):
        self.black_box_model = black_box_model
        self.train_data = list()
        self.disc_data = list()
        self.test_data = list()
        self.protected_list = [self.black_box_model.feature_list[i] for i in protected_list] # ["age"]
        self.protected_list_no = protected_list
        self.no_train_data_sample = no_train_data_sample
        self.no_test = 0
        self.no_disc = 0
        self.real_time_consumed = 0
        self.cpu_time_consumed = 0
        self.protected_value_comb = self.generate_protected_value_combination()
        self.no_prot = len(protected_list)
        if show_logging:
            logging.basicConfig(format="", level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def generate_protected_value_combination(self):
        """Generate all combinations of protected-attribute values."""
        res = list()
        for index_protected in self.protected_list_no:
            MinMax = self.black_box_model.data_range[index_protected]
            res.append(list(range(MinMax[0], MinMax[1] + 1)))
        return list(itertools.product(*res))

    def create_train_data(self, num):
        """Generate random training samples and label them using the black-box model."""
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
        """Train a decision tree that approximates the black-box model."""
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

        logging.info(f"Starting fairness test -- {label[0]}")
           
        # Extract metadata from the experiment label.
        _, _, dataset_name, _, _ = label[0].split("-")
        NUM_BINS = 5
        
        def bin_dataset_summary(dataset_summary):
            """Convert numerical attributes in the dataset summary into categorical bins."""
            binned_dataset_summary = copy.deepcopy(dataset_summary)
            
            for _, attr_info in binned_dataset_summary.items():
                if attr_info.get("type") == "numerical":
                    start, end = map(int, attr_info["range"])
                    bin_edges = np.linspace(start, end, NUM_BINS + 1, dtype=int)

                    categories = [
                        f"{bin_edges[i]}~{bin_edges[i+1]-1}" if i < len(bin_edges)-2 else f"{bin_edges[i]}~{bin_edges[i+1]}"
                        for i in range(len(bin_edges) - 1)
                    ] 

                    attr_info["type"] = "categorical"
                    attr_info["bins"] = categories
                    attr_info["range"] = [0, len(categories) - 1]
                    
            return binned_dataset_summary
                
        
        def binned_instance(instance, binned_dataset_summary):
            """
            Convert a single instance into its binned representation.
            Only numerical attributes are transformed.
            """
            summary_list = [{"name": k, **v} for k, v in binned_dataset_summary.items()]
            summary_list = summary_list[:-1]
            converted_instance = instance.copy()
            
            for i, info in enumerate(summary_list):
                if "bins" in info:
                    value = int(instance[i])
                    for idx, bin_label in enumerate(info["bins"]):
                        lower, upper = map(int, bin_label.split("~"))
                        if lower <= value <= upper:
                            converted_instance[i] = idx
                            break
                        
            return converted_instance
        
       
        def is_valid(inp, occur_table):
            """
            Check whether an input instance is valid based on pairwise occurrence constraints.
            The input must not include the class label.
            """
            binned_inp = binned_instance(inp, binned_ds_summary)
            pairs = list(combinations(range(len(binned_inp)), 2))
            
            for idx, (i, j) in enumerate(pairs):
                key = (binned_inp[i], binned_inp[j])
                if key not in occur_table[idx] or occur_table[idx][key]==0:
                    return False 
                       
            return True
        
        ds_summary = getattr(dataset_config, dataset_name)
        binned_ds_summary = bin_dataset_summary(ds_summary)

        # Load the occurrence table used for validity checking.
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
            
            retrain_dataset_size = int(load_train_size(dataset_name) * 0.1)
            
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
                    
                    # Split newly found IDIs into valid and invalid pairs.
                    pair_raw_IDIs = [
                        [new_disc_data[i], new_disc_data[i + 1]]
                        for i in range(0, len(new_disc_data), 2)
                    ]
                    
                    for IDI_1, IDI_2 in pair_raw_IDIs:
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

        logging.info(f"The fairness test is completed")

        # Save all detected discriminatory instances.
        raw_IDIs_path = f"{root_dir}/IDIs/raw/{label[0]}.csv"
        logging.info(f"Saving the detected discriminatory instances to {raw_IDIs_path}")
        with open(raw_IDIs_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.disc_data)
        
        # Save valid discriminatory instances.
        valid_IDIs_path = f"{root_dir}/IDIs/valid/{label[0]}.csv"
        logging.info(f"Saving the valid discriminatory instances to {valid_IDIs_path}")
        with open(valid_IDIs_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(valid_IDIs)
        
        # Save invalid discriminatory instances.
        invalid_IDIs_path = f"{root_dir}/IDIs/invalid/{label[0]}.csv"
        logging.info(f"Saving the invalid discriminatory instances to {invalid_IDIs_path}")
        with open(invalid_IDIs_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(invalid_IDIs)
            
            