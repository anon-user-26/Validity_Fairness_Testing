import os, sys, csv, copy
from joblib import load, dump
import pandas as pd
import random
import numpy as np
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from datasets_original import dataset_config

# 準備

model_name = sys.argv[1]
dataset_name = sys.argv[2]
protected_name = sys.argv[3]
method = sys.argv[4]
runtime = sys.argv[5]




# 訓練データを読み込む
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = f"{root_dir}/datasets_prepared/train/{dataset_name}_train.csv"
train_df = pd.read_csv(train_path)
train_dataset = train_df.values
X = train_dataset[:, :-1]
y = train_dataset[:, -1]


# valid IDIs, invalid IDIsを読み込む
with open(f'{root_dir}/IDIs/valid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    valid_IDIs = [list(map(int, row)) for row in csvreader]
with open(f'{root_dir}/IDIs/invalid/{method}-{model_name}-{dataset_name}-{protected_name}-{runtime}.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    invalid_IDIs = [list(map(int, row)) for row in csvreader]

# 再学習に必要な数の valid IDIs, invalid IDIs がなければ，再学習を行わない．訓練済みモデルの評価結果も残さない．
if len(valid_IDIs) < len(train_dataset)*0.1 or len(invalid_IDIs) < len(train_dataset)*0.1:
    print("Not enough IDIs. Skip the experiment.")
    sys.exit(1) 


# 他2つのモデルを学習（majority votingの準備）
RanForest = RandomForestClassifier(n_estimators=50, criterion="gini")
MLP = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), activation="relu", solver="adam", learning_rate="adaptive",
                    max_iter=5000, early_stopping=True)   
SVM = LinearSVC(penalty="l2", dual="auto")
if model_name == "RanForest":
    other_models = [MLP, SVM]
elif model_name == "MLP":
    other_models = [RanForest, SVM]
elif model_name == "SVM":
    other_models = [RanForest, MLP]
else:
    print(f"no ML algorithm called {model_name}.")
for model in other_models:
    model.fit(X,y)



def majority_voting(inp1_features, inp2_features):
    votes = []
    # 各モデルについて、2つのデータ両方の予測を追加（合計4票）
    for model in other_models:
        votes.append(int(model.predict([inp1_features])[0]))
        votes.append(int(model.predict([inp2_features])[0]))
    # 多数決
    counter = Counter(votes)
    max_count = max(counter.values())
    # 最頻値候補（複数あればすべて入る）
    modes = [k for k, v in counter.items() if v == max_count]
    # 複数ある場合はランダムに1つ選択
    label = random.choice(modes)
    return label

# 評価の準備：再学習用データの多様性
# check L0_distance
def L0_distance(inp1,inp2):
    L0_distance = 0
    for i in range(len(inp1)):
        if inp1[i]!=inp2[i]:
            L0_distance += 1
    return L0_distance

# check L1_distance
def L1_distance(inp1,inp2):
    L1_distance = 0
    for i in range(len(inp1)):     
        difference = abs(inp1[i]-inp2[i])
        L1_distance += float(difference)/(input_bounds[i][1] - input_bounds[i][0])
    return L1_distance

# check binned_L0_distance
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

NUM_BINS = 5

ds_summary = getattr(dataset_config, dataset_name)
input_bounds = [v["range"] for k, v in ds_summary.items() if v["type"] != "output"]

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

binned_ds_summary = bin_dataset_summary(ds_summary)

def binned_L0_distance(inp1,inp2):
    binned_inp1 = binned_instance(inp1, binned_ds_summary)
    binned_inp2 = binned_instance(inp2, binned_ds_summary)
    binned_L1_distance = 0
    for i in range(len(inp1)):
        if binned_inp1[i]!=binned_inp2[i]:
            binned_L1_distance += 1
    return binned_L1_distance

def average_pairwise_distance(dataset, distance_fn):
    pairwise_distance = 0
    count=0
    dataset_num = len(dataset)
    for i in range(0,dataset_num-1):
        for j in range(i+1,dataset_num):           
            pairwise_distance += distance_fn(dataset[i], dataset[j])         
            count += 1
    return pairwise_distance / count


# 評価の準備：モデルの精度
# テストデータを読み込む
test_accuracy_path = f"{root_dir}/datasets_prepared/test_accuracy/{dataset_name}_test.csv"
test_df = pd.read_csv(test_accuracy_path)
test_dataset = test_df.values

def test_accuracy(model,test_dataset):
    test_dataset_num = len(test_dataset)
    correct_num = 0
    for test_data in test_dataset:
        test_x = test_data[:-1]
        test_y = test_data[-1]
        test_x = np.reshape(test_x, (1, -1))
        if model.predict(test_x)[0] == test_y:
            correct_num += 1
    accuracy = float(correct_num)/test_dataset_num
    return accuracy

# 評価の準備：モデルの公平性
# test_IFr_set, test_valid_IFr_setを，ブロックとして読み込む
def load_blocks(path):
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

# 評価の準備，再学習済みモデルの IFr, valid_IFr を測定
def test_IFr(model, test_IFr_set):
    diff_blocks = 0
    total_blocks = len(test_IFr_set)
    for block in test_IFr_set:
        outputs = [model.predict([input])[0] for input in block]
        # このブロック内で1つでも出力が違うかどうかをチェック
        if len(set(outputs)) > 1:
            diff_blocks += 1
    return diff_blocks / total_blocks


# 評価：訓練済みモデル
models_trained_path = f"{root_dir}/models_trained/{model_name}_{dataset_name}.joblib"
trained_model = load(models_trained_path)
accuracy_trained = test_accuracy(trained_model,test_dataset)
IFr_trained = test_IFr(trained_model, test_IFr_set)
valid_IFr_trained = test_IFr(trained_model, test_valid_IFr_set)





def retrain_CuT(model_name, dataset_name, protected_name, validity, save_to="CuT.joblib"):
    
    # valid_IDIs. invalid_IDIsから，ランダムにサンプリング
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
    
    # majority votingにより，added_pairsの各ペアに同一ラベルを付与し，再学習用データセットを構築
    added_dataset = []
    for (inp1, inp2) in added_pairs:
        
        inp1_features = inp1[:-1]
        inp2_features = inp2[:-1]
        
        label = majority_voting(inp1_features, inp2_features)
        
        new_inp1 = inp1_features + [label]
        new_inp2 = inp2_features + [label]
        added_dataset.append(new_inp1)
        added_dataset.append(new_inp2)
    
    # 訓練データ，valid_IDIS, invalid_IDIsをくっつけて，retrain_dataset（X,y）をつくる
    
    added_dataset = np.array(added_dataset)
    retrain_dataset = np.vstack([train_dataset, added_dataset])
    np.random.shuffle(retrain_dataset)
    
    X_retrain = retrain_dataset[:, :-1]
    y_retrain = retrain_dataset[:, -1]

    # retrain CuT based on train data
    CuT = None
    if model_name == "RanForest":
        CuT = RandomForestClassifier(n_estimators=50, criterion="gini")
        CuT.fit(X_retrain, y_retrain)
    elif model_name == "MLP":
        CuT = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), activation="relu", solver="adam", learning_rate="adaptive"
                            ,max_iter=5000, early_stopping=True)
        CuT.fit(X_retrain, y_retrain)
    elif model_name == "SVM":
        CuT = LinearSVC(penalty="l2", dual="auto")
        CuT.fit(X_retrain, y_retrain)
    else:
        print(f"no ML algorithm called {model_name}.")
    
    # save CuT
    if save_to is not None:
        dump(CuT, save_to)
    
    
    # 評価, 用いた再学習用データ（追加分，ラベルなし，片側だけ）の多様性を測定
    added_single_instances = [pair[0][:-1] for pair in added_pairs]
    average_pairwise_L0_distance = average_pairwise_distance(added_single_instances, L0_distance)
    average_pairwise_L1_distance = average_pairwise_distance(added_single_instances, L1_distance)
    average_pairwise_binned_L0_distance = average_pairwise_distance(added_single_instances, binned_L0_distance)
    
    
    # 評価，再学習済みモデルの精度を測定
    accuracy_retrained = test_accuracy(CuT,test_dataset)
    
    
    # 評価，再学習済みモデルの公平性を測定
    IFr_retrained = test_IFr(CuT, test_IFr_set)
    valid_IFr_retrained = test_IFr(CuT, test_valid_IFr_set)
    
    
    # results/に結果をtxtで記録
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
        if not content.strip():  # 空ファイルならヘッダーを書く
            myfile.write("validity average_pairwise_L0_distance average_pairwise_L1_distance average_pairwise_binned_L0_distance accuracy_trained IFr_trained valid_IFr_trained accuracy_retrained IFr_retrained valid_IFr_retrained\n")
        myfile.write(f"{validity} {average_pairwise_L0_distance} {average_pairwise_L1_distance} {average_pairwise_binned_L0_distance} {accuracy_trained} {IFr_trained} {valid_IFr_trained} {accuracy_retrained} {IFr_retrained} {valid_IFr_retrained}\n")
    print("Results saved to: {}".format(save_path))

    return CuT




# 妥当性のレベルを指定して，再学習を実行 
models_retrained_path = f"{root_dir}/models_retrained/{model_name}_{dataset_name}_{protected_name}.joblib"

for validity in [i / 100 for i in range(0, 101, 5)]:
    retrain_CuT(model_name, dataset_name, protected_name, validity, models_retrained_path)
