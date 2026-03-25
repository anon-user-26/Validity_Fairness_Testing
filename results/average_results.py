import sys, os
import numpy as np

def average(model_name, dataset_name, protected_name, validiy):

    validity_list = []
    average_pairwise_L0_distance_list = []
    average_pairwise_L1_distance_list = []
    average_pairwise_binned_L0_distance_list = []
    accuracy_trained_list = []
    IFr_trained_list = []
    valid_IFr_trained_list = []
    accuracy_retrained_list = []
    IFr_retrained_list = []
    valid_IFr_retrained_list = []

    result_path = f"{model_name}/{dataset_name}/{protected_name}/{model_name}_{dataset_name}_{protected_name}_{validiy}.txt"
    with open(result_path) as f:
        lines = f.readlines()[1:]  # Skip header line
        for line in lines:
            validity,average_pairwise_L0_distance,average_pairwise_L1_distance,average_pairwise_binned_L0_distance,accuracy_trained,IFr_trained,valid_IFr_trained,accuracy_retrained,IFr_retrained,valid_IFr_retrained = line.split()
            
            validity_list.append(float(validity))
            average_pairwise_L0_distance_list.append(float(average_pairwise_L0_distance))
            average_pairwise_L1_distance_list.append(float(average_pairwise_L1_distance))
            average_pairwise_binned_L0_distance_list.append(float(average_pairwise_binned_L0_distance))
            accuracy_trained_list.append(float(accuracy_trained))
            IFr_trained_list.append(float(IFr_trained))
            valid_IFr_trained_list.append(float(valid_IFr_trained))
            accuracy_retrained_list.append(float(accuracy_retrained))
            IFr_retrained_list.append(float(IFr_retrained))
            valid_IFr_retrained_list.append(float(valid_IFr_retrained))

    
    mean = np.mean(validity_list)
    std = np.std(validity_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    a = f"validity: {lower} {mean} {upper}"
    
    mean = np.mean(average_pairwise_L0_distance_list)
    std = np.std(average_pairwise_L0_distance_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    b = f"average_pairwise_L0_distance: {lower} {mean} {upper}"
    
    mean = np.mean(average_pairwise_L1_distance_list)
    std = np.std(average_pairwise_L1_distance_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    c = f"average_pairwise_L1_distance: {lower} {mean} {upper}"
    
    mean = np.mean(average_pairwise_binned_L0_distance_list)
    std = np.std(average_pairwise_binned_L0_distance_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    d = f"average_pairwise_binned_L0_distance: {lower} {mean} {upper}"
    
    mean = np.mean(accuracy_trained_list)
    std = np.std(accuracy_trained_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    e = f"accuracy_trained: {lower} {mean} {upper}"
    
    mean = np.mean(IFr_trained_list)
    std = np.std(IFr_trained_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    f = f"IFr_trained: {lower} {mean} {upper}"
    
    mean = np.mean(valid_IFr_trained_list)
    std = np.std(valid_IFr_trained_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    g = f"valid_IFr_trained: {lower} {mean} {upper}"
    
    mean = np.mean(accuracy_retrained_list)
    std = np.std(accuracy_retrained_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    h = f"accuracy_retrained: {lower} {mean} {upper}"
    
    mean = np.mean(IFr_retrained_list)
    std = np.std(IFr_retrained_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    i = f"IFr_retrained: {lower} {mean} {upper}"
    
    mean = np.mean(valid_IFr_retrained_list)
    std = np.std(valid_IFr_retrained_list)
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    j = f"valid_IFr_retrained: {lower} {mean} {upper}"
    
    with open(f"{model_name}/{dataset_name}/{protected_name}/{model_name}_{dataset_name}_{protected_name}_{validiy}_average.txt", "w") as myfile:
        myfile.write(a + "\n" + b + "\n" + c + "\n" + d + "\n" + e + "\n" + f + "\n" + g + "\n" + h + "\n" + i + "\n" + j + "\n")
                        
if __name__ == '__main__':
    file = sys.argv[1]                      # SVM_Credit_age_0.05.txt
    filename = os.path.splitext(file)[0]    # SVM_Credit_age_0.05

    model_name, dataset_name, protected_name, validity = filename.split("_")

    average(model_name, dataset_name, protected_name, validity)
