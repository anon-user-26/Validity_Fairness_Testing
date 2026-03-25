import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_name = sys.argv[1]


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_original_path = f"{root_dir}/datasets_original"

df = pd.read_csv(f"{dataset_original_path}/{dataset_name}.csv")

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)

# train data, test dataをcsvファイルとして保存
train_df.to_csv(f"{root_dir}/datasets_prepared/train/{dataset_name}_train.csv", index=False)
test_df.to_csv(f"{root_dir}/datasets_prepared/test_accuracy/{dataset_name}_test.csv", index=False)
