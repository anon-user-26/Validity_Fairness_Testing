import subprocess
import os

execution_times = 1
runtime = 3600

root_dir = os.path.dirname(os.path.abspath(__file__))

scenarios = [
    #("SVM",       "Adult",  "age"),

    #("SVM",       "Credit", "age"),
    
    #("SVM",       "Credit", "sex"),
    
    #("MLP",       "Adult",  "age"),
    
    ("MLP",       "Adult",  "race"),
    
    # ("MLP",       "Adult",  "sex"),
    
    # ("MLP",       "Bank",   "age"),
    
    #("MLP",       "Credit", "age"),
    
    #("MLP",       "Credit", "sex"),

    #("RanForest", "Bank",   "age"),
    
    #("RanForest", "Credit", "age"),
    
    #("RanForest", "Credit", "sex"),
]

for i in range(execution_times):
    for model_name, dataset_name, protected_name in scenarios:

        split_train_test_path = f"{root_dir}/datasets_prepared/split_train_test.py"
        subprocess.call(f"python {split_train_test_path} {dataset_name}", shell=True)
        
        make_occ_table_path = f"{root_dir}/datasets_prepared/make_occ_table.py"
        subprocess.call(f"python {make_occ_table_path} {dataset_name}", shell=True)
        
        make_test_IFr_path = f"{root_dir}/datasets_prepared/make_test_IFr.py"
        subprocess.call(f"python {make_test_IFr_path} {dataset_name} {protected_name}", shell=True)
        
        make_test_valid_IFr_path = f"{root_dir}/datasets_prepared/make_test_valid_IFr.py"
        subprocess.call(f"python {make_test_valid_IFr_path} {dataset_name} {protected_name}", shell=True)
        
        train_path = f"{root_dir}/models_trained/train.py"
        subprocess.call(f"python {train_path} {model_name} {dataset_name} {protected_name}", shell=True)
        
        exp_path = f"{root_dir}/exp.py"
        subprocess.call(f"python {exp_path} --dataset_name {dataset_name} --protected_attr {protected_name} --model_name {model_name} --method aft --runtime {runtime} --repeat 1", shell=True)

        retrain_path = f"{root_dir}/models_retrained/retrain.py"
        subprocess.call(f"python {retrain_path} {model_name} {dataset_name} {protected_name} aft {runtime}", shell=True)
        
