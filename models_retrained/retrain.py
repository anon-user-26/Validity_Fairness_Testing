import copy
import csv
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from datasets_original import dataset_config


NUM_BINS = 5


model_name = sys.argv[1]
dataset_name = sys.argv[2]
protected_name = sys.argv[3]
method = sys.argv[4]
runtime = sys.argv[5]


train_path = f"{root_dir}/datasets_prepared/train/{dataset_name}_train.csv"
train_df = pd.read_csv(train_path)
train_dataset = train_df.values
X = train_dataset[:, :-1]
y = train_dataset[:, -1]


with open(f'{root_dir}/IDIs/valid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    valid_IDIs = [list(map(int, row)) for row in csvreader]
with open(f'{root_dir}/IDIs/invalid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    invalid_IDIs = [list(map(int, row)) for row in csvreader]

# Skip the experiment if there are not enough valid/invalid IDIs for retraining.
if len(valid_IDIs) < len(train_dataset)*0.1 or len(invalid_IDIs) < len(train_dataset)*0.1:
    print("Not enough IDIs. Skip the experiment.")
    sys.exit(1) 


# Train auxiliary models for majority voting
svm_model  = LinearSVC(penalty="l2", dual="auto")
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), activation="relu", solver="adam", learning_rate="adaptive",
                    max_iter=5000, early_stopping=True)   
ranforest_model  = RandomForestClassifier(n_estimators=50, criterion="gini")
if model_name == "SVM":
    other_models = [mlp_model, ranforest_model]
elif model_name == "MLP":
    other_models = [svm_model, ranforest_model]
elif model_name == "RanForest":
    other_models = [svm_model, mlp_model]
else:
    print(f"No ML algorithm called {model_name}.")
for model in other_models:
    model.fit(X,y)


ds_summary = getattr(dataset_config, dataset_name)
input_bounds = [v["range"] for k, v in ds_summary.items() if v["type"] != "output"]


def majority_voting(inp1_features, inp2_features):
    """Assign one label to an IDI pair by majority voting of auxiliary models."""
    votes = []
    
    for model in other_models:
        votes.append(int(model.predict([inp1_features])[0]))
        votes.append(int(model.predict([inp2_features])[0]))
    
    counter = Counter(votes)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]
    # Select one label uniformly at random in case of ties
    label = random.choice(modes)
    
    return label


def L0_distance(inp1, inp2):
    """Compute the L0 distance between two instances."""
    distance = 0
    for i in range(len(inp1)):
        if inp1[i] != inp2[i]:
            distance += 1
    return distance


def L1_distance(inp1, inp2):
    """Compute the normalized L1 distance between two instances."""
    distance = 0
    for i in range(len(inp1)):
        difference = abs(inp1[i] - inp2[i])
        distance += float(difference) / (input_bounds[i][1] - input_bounds[i][0])
    return distance


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
    """Create a binned version of the dataset summary for Binned-L0 computation."""
    
    binned_summary = copy.deepcopy(dataset_summary)

    for attr, attr_info in binned_summary.items():
        if attr_info.get("type") == "numerical":
            start, end = map(int, attr_info["range"])
            bin_edges = np.linspace(start, end, NUM_BINS + 1, dtype=int)
  
            bins = [
                f"{bin_edges[i]}~{bin_edges[i+1] - 1}" if i < len(bin_edges) - 2 else f"{bin_edges[i]}~{bin_edges[i+1]}"
                for i in range(len(bin_edges) - 1)
            ] 

            attr_info["type"] = "categorical"
            attr_info["bins"] = bins
            attr_info["range"] = [0, len(bins) - 1]

    return binned_summary


binned_ds_summary = bin_dataset_summary(ds_summary)


def binned_L0_distance(inp1, inp2):
    """Compute the Binned-L0 distance between two instances."""
    binned_inp1 = binned_instance(inp1, binned_ds_summary)
    binned_inp2 = binned_instance(inp2, binned_ds_summary)

    distance = 0
    for i in range(len(inp1)):
        if binned_inp1[i] != binned_inp2[i]:
            distance += 1

    return distance


def average_pairwise_distance(dataset, distance_fn):
    """Compute the average pairwise distance over all instance pairs."""
    total_distance = 0
    count = 0
    dataset_num = len(dataset)

    for i in range(dataset_num - 1):
        for j in range(i + 1, dataset_num):
            total_distance += distance_fn(dataset[i], dataset[j])
            count += 1

    return total_distance / count


test_accuracy_path = f"{root_dir}/datasets_prepared/test_accuracy/{dataset_name}_test.csv"
test_df = pd.read_csv(test_accuracy_path)
test_dataset = test_df.values


def test_accuracy(model, test_dataset):
    """Evaluate classification accuracy on the test set."""
    test_dataset_num = len(test_dataset)
    correct_num = 0

    for test_data in test_dataset:
        test_x = test_data[:-1]
        test_y = test_data[-1]
        test_x = np.reshape(test_x, (1, -1))

        if model.predict(test_x)[0] == test_y:
            correct_num += 1

    return float(correct_num) / test_dataset_num


def load_blocks(path):
    """Load block-structured test inputs separated by blank lines."""
    blocks = []
    block = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "":
                if block:
                    blocks.append(block)
                    block = []
                continue
            block.append([int(x) for x in line.split(",")])

    if block:
        blocks.append(block)

    return blocks


test_IFr_set_path = f"{root_dir}/datasets_prepared/test_IFr/{dataset_name}_{protected_name}_test_IFr_set.csv"
test_valid_IFr_set_path = f"{root_dir}/datasets_prepared/test_valid_IFr/{dataset_name}_{protected_name}_test_valid_IFr_set.csv"

test_IFr_set = load_blocks(test_IFr_set_path)
test_valid_IFr_set = load_blocks(test_valid_IFr_set_path)


def test_IFr(model, test_IFr_set):
    """Evaluate IFr as the ratio of blocks with inconsistent predictions."""
    diff_blocks = 0
    total_blocks = len(test_IFr_set)

    for block in test_IFr_set:
        outputs = [model.predict([instance])[0] for instance in block]
        if len(set(outputs)) > 1:
            diff_blocks += 1

    return diff_blocks / total_blocks


models_trained_path = f"{root_dir}/models_trained/{model_name}_{dataset_name}.joblib"
trained_model = load(models_trained_path)

accuracy_trained = test_accuracy(trained_model,test_dataset)
IFr_trained = test_IFr(trained_model, test_IFr_set)
valid_IFr_trained = test_IFr(trained_model, test_valid_IFr_set)


def retrain_CuT(model_name, dataset_name, protected_name, validity, save_to="CuT.joblib"):
    """Retrain the classifier under test with a validity-controlled IDI set."""
    
    def pair_list(lst):
        return [lst[i:i+2] for i in range(0, len(lst), 2)]
    
    valid_pairs = pair_list(valid_IDIs)
    invalid_pairs = pair_list(invalid_IDIs)

    added_pairs_num = int(len(train_dataset) * 0.1 * 0.5)
    added_valid_pairs_num = int(added_pairs_num * validity)
    added_invalid_pairs_num = added_pairs_num - added_valid_pairs_num
    
    added_valid_pairs = random.sample(valid_pairs, added_valid_pairs_num)
    added_invalid_pairs = random.sample(invalid_pairs, added_invalid_pairs_num)
    added_pairs = added_valid_pairs + added_invalid_pairs
    
    # Build the retraining set by assigning one shared label to each IDI pair.
    added_dataset = []
    for inp1, inp2 in added_pairs:
        inp1_features = inp1[:-1]
        inp2_features = inp2[:-1]
        
        label = majority_voting(inp1_features, inp2_features)
        
        new_inp1 = inp1_features + [label]
        new_inp2 = inp2_features + [label]
        added_dataset.append(new_inp1)
        added_dataset.append(new_inp2)
    
    added_dataset = np.array(added_dataset)
    retrain_dataset = np.vstack([train_dataset, added_dataset])
    np.random.shuffle(retrain_dataset)
    
    X_retrain = retrain_dataset[:, :-1]
    y_retrain = retrain_dataset[:, -1]

    CuT = None
    if model_name == "SVM":
        CuT = LinearSVC(penalty="l2", dual="auto")
    elif model_name == "MLP":
        CuT = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), activation="relu", solver="adam", learning_rate="adaptive"
                            ,max_iter=5000, early_stopping=True)
    elif model_name == "RanForest":
        CuT = RandomForestClassifier(n_estimators=50, criterion="gini")
    else:
        print(f"no ML algorithm called {model_name}.")
        return None
    
    CuT.fit(X_retrain, y_retrain)
    
    if save_to is not None:
        dump(CuT, save_to)
    
    
    # 評価, 用いた再学習用データ（追加分，ラベルなし，片側だけ）の多様性を測定
    # Measure the diversity of the added instances using one representative per pair.
    added_single_instances = [pair[0][:-1] for pair in added_pairs]
    average_pairwise_L0_distance = average_pairwise_distance(added_single_instances, L0_distance)
    average_pairwise_L1_distance = average_pairwise_distance(added_single_instances, L1_distance)
    average_pairwise_binned_L0_distance = average_pairwise_distance(added_single_instances, binned_L0_distance)
    
    accuracy_retrained = test_accuracy(CuT,test_dataset)
    IFr_retrained = test_IFr(CuT, test_IFr_set)
    valid_IFr_retrained = test_IFr(CuT, test_valid_IFr_set)
    
    print(f"validity: {validity}")
    print(f"average_pairwise_L0_distance: {average_pairwise_L0_distance:.6f}")
    print(f"average_pairwise_L1_distance: {average_pairwise_L1_distance:.6f}")
    print(f"average_pairwise_binned_L0_distance: {average_pairwise_binned_L0_distance:.6f}")
    print(f"Before retrain - Accuracy: {accuracy_trained:.6f}")
    print(f"Before retrain - Fairness (IFr): {IFr_trained:.6f}")
    print(f"Before retrain - Fairness (valid IFr): {valid_IFr_trained:.6f}")
    print(f"After retrain - Accuracy: {accuracy_retrained:.6f}")
    print(f"After retrain - Fairness (IFr): {IFr_retrained:.6f}")
    print(f"After retrain - Fairness (valid IFr): {valid_IFr_retrained:.6f}")
    
    save_path = f"{root_dir}/results/{model_name}/{dataset_name}/{protected_name}/{model_name}_{dataset_name}_{protected_name}_{validity}.txt"
    
    with open(save_path, "a+") as myfile:
        myfile.seek(0)
        content = myfile.read()
        
        if not content.strip():
            myfile.write(
                "validity "
                "average_pairwise_L0_distance "
                "average_pairwise_L1_distance "
                "average_pairwise_binned_L0_distance "
                "accuracy_trained "
                "IFr_trained "
                "valid_IFr_trained "
                "accuracy_retrained "
                "IFr_retrained "
                "valid_IFr_retrained\n"
            )
            
        myfile.write(
            f"{validity} "
            f"{average_pairwise_L0_distance} "
            f"{average_pairwise_L1_distance} "
            f"{average_pairwise_binned_L0_distance} "
            f"{accuracy_trained} "
            f"{IFr_trained} "
            f"{valid_IFr_trained} "
            f"{accuracy_retrained} "
            f"{IFr_retrained} "
            f"{valid_IFr_retrained}\n"
        )
    
    print(f"Results saved to: {save_path}")

    return CuT

models_retrained_path = f"{root_dir}/models_retrained/{model_name}_{dataset_name}_{protected_name}.joblib"

for validity in [i / 100 for i in range(0, 101, 5)]:
    retrain_CuT(model_name, dataset_name, protected_name, validity, models_retrained_path)
