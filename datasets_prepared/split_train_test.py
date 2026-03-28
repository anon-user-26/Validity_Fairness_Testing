import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split


# Argument
dataset_name = sys.argv[1]

# Path configuration
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(root_dir, "datasets_original", f"{dataset_name}.csv")

train_output_path = os.path.join(root_dir, "datasets_prepared", "train", f"{dataset_name}_train.csv")
test_output_path = os.path.join(root_dir, "datasets_prepared", "test_accuracy", f"{dataset_name}_test.csv")

# Load dataset
df = pd.read_csv(input_path)

# Train-test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    shuffle=True,
)

# Save outputs
train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)
