import average_results
import os

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".txt"):
            filename = os.path.splitext(file)[0]  # "SVM_Credit_age_0.05"
            parts = filename.split("_")  # ['SVM', 'Credit', 'age', '0.05']

            if len(parts) != 4:
                continue

            model_name = parts[0]       # SVM
            dataset_name = parts[1]     # Credit
            protected_name = parts[2]   # age
            validiy = parts[3]          # 0.05

            average_results.average(model_name, dataset_name, protected_name, validiy)

