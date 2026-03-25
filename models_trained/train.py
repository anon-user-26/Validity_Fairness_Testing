import os, sys
from joblib import load, dump
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def train_CuT(dataset_name, model_name, protected_name, save_to="CuT.joblib"):
    
    # 訓練データを読み込む
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = f"{root_dir}/datasets_prepared/train/{dataset_name}_train.csv"
    train_df = pd.read_csv(train_path)
    train_dataset = train_df.values
    X = train_dataset[:, :-1]
    y = train_dataset[:, -1]

    # train CuT based on train data
    CuT = None
    if model_name == "DecTree":
        CuT = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None)
        CuT.fit(X, y)
    elif model_name == "RanForest":
        CuT = RandomForestClassifier(n_estimators=50, criterion="gini")
        # CuT = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        #                             max_depth=5, max_features='auto', max_leaf_nodes=None,
        #                             min_impurity_decrease=0.0, min_impurity_split=None,
        #                             min_samples_leaf=1, min_samples_split=2,
        #                             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
        #                             oob_score=False, random_state=42, verbose=0,
        #                             warm_start=False)
        CuT.fit(X, y)
    elif model_name == "LogReg":
        CuT = LogisticRegression(penalty="l2")
        CuT.fit(X, y)
    elif model_name == "NB":
        CuT = CategoricalNB(alpha=1.0)
        CuT.fit(X, y)
    elif model_name == "MLP":
        CuT = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), activation="relu", solver="adam", learning_rate="adaptive",
                            max_iter=5000, early_stopping=True)
        # CuT = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), activation='relu', solver='adam',
        #                     alpha=0.0001, batch_size='auto', learning_rate='adaptive',
        #                     learning_rate_init=0.001, power_t=0.5, max_iter=5000,
        #                     shuffle=True, random_state=42, tol=0.0001,
        #                     verbose=False, warm_start=False, momentum=0.9,
        #                     nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1,
        #                     beta_1=0.9, beta_2=0.999, epsilon=1e-08)   
        CuT.fit(X, y)
    elif model_name == "Adaboost":
        CuT = AdaBoostClassifier(n_estimators=100, algorithm="SAMME")
        CuT.fit(X, y)
    elif model_name == "GBDT":
        CuT = GradientBoostingClassifier(n_estimators=100)
        CuT.fit(X, y)
    elif model_name == "SVM":
        CuT = LinearSVC(penalty="l2", dual="auto")
        #CuT = SVC(gamma=0.0025)
        CuT.fit(X, y)
    else:
        print(f"no ML algorithm called {model_name}.")

    # save CuT
    if save_to is not None:
        dump(CuT, save_to)
    
    
    # # 評価，学習済みモデルの精度を測定
    
    # テストデータを読み込む
    # test_accuracy_path = f"{root_dir}/datasets_prepared/test_accuracy/{dataset_name}_test.csv"
    # test_df = pd.read_csv(test_accuracy_path)
    # test_dataset = test_df.values
    
    # def test_accuracy(model,test_dataset):
    #     test_dataset_num = len(test_dataset)
    #     correct_num = 0
    #     for test_data in test_dataset:
    #         test_x = test_data[:-1]
    #         test_y = test_data[-1]
    #         test_x = np.reshape(test_x, (1, -1))
    #         if model.predict(test_x)[0] == test_y:
    #             correct_num += 1
    #     accuracy = float(correct_num)/test_dataset_num
    #     return accuracy
    
    # accuracy = test_accuracy(CuT,test_dataset)
    # print(accuracy)
    
    
    # # 評価
    # # test_IFr_set, test_valid_IFr_setを，ブロックとして読み込む
    # def load_blocks(path):
    #     blocks = []
    #     block = []
    #     with open(path) as f:
    #         for line in f:
    #             line = line.strip()
    #             if line == "":
    #                 if block:
    #                     blocks.append(block)
    #                     block = []
    #                 continue
    #             block.append([int(x) for x in line.split(",")])
    #     if block:
    #         blocks.append(block)
    #     return blocks

    # test_IFr_set_path = f"{root_dir}/datasets_prepared/test_IFr/{dataset_name}_{protected_name}_test_IFr_set.csv"
    # test_valid_IFr_set_path = f"{root_dir}/datasets_prepared/test_valid_IFr/{dataset_name}_{protected_name}_test_valid_IFr_set.csv"
    # test_IFr_set = load_blocks(test_IFr_set_path)
    # test_valid_IFr_set = load_blocks(test_valid_IFr_set_path)
    
    # # 評価，再学習済みモデルの IFr, valid_IFr を測定
    # def test_IFr(model, test_IFr_set):
    #     diff_blocks = 0
    #     total_blocks = len(test_IFr_set)
    #     for block in test_IFr_set:
    #         outputs = [model.predict([input])[0] for input in block]
    #         # このブロック内で1つでも出力が違うかどうかをチェック
    #         if len(set(outputs)) > 1:
    #             diff_blocks += 1
    #     return diff_blocks / total_blocks

    # IFr = test_IFr(CuT, test_IFr_set)
    # valid_IFr = test_IFr(CuT, test_valid_IFr_set)

    
    # # モデルの評価結果をresults/models_trainedにtxtで記録
    
    # print(f"Before retrain - Accuracy: {accuracy:.6f}")
    # print(f"Before retrain - Fairness (IFr): {IFr:.6f}")
    # print(f"Before retrain - Fairness (valid IFr): {valid_IFr:.6f}")
    
    # save_path = f"{root_dir}/results/models_trained/{model_name}_{dataset_name}_{protected_name}.txt"
    # with open(save_path, "a+") as myfile:
    #     myfile.seek(0)
    #     content = myfile.read()
    #     if not content.strip():  # 空ファイルならヘッダーを書く
    #         myfile.write("accuracy IFr valid_IFr\n")
    #     myfile.write(f"{accuracy} {IFr} {valid_IFr}\n")
    # print("Results saved to: {}".format(save_path))

    return CuT

    

model_name = sys.argv[1]
dataset_name = sys.argv[2]
protected_name = sys.argv[3]

current_dir = os.path.dirname(os.path.abspath(__file__))
model_trained_path = f"{current_dir}/{model_name}_{dataset_name}.joblib"

train_CuT(dataset_name, model_name, protected_name, model_trained_path)
