import os
from analysis import analysis  # Import your analysis() function


def analysis_all():
    """
    Automatically run analysis(model_name, dataset_name, protected_name)
    for every directory structure: ./model/dataset/protected/
    """

    # Loop over all top-level directories (model names)
    for model_name in os.listdir("."):
        model_path = os.path.join(".", model_name)

        # Skip if not a directory
        if not os.path.isdir(model_path):
            continue

        # Loop over dataset directories
        for dataset_name in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_name)

            # Skip if not a directory
            if not os.path.isdir(dataset_path):
                continue

            # Loop over protected attribute directories
            for protected_name in os.listdir(dataset_path):
                protected_path = os.path.join(dataset_path, protected_name)

                # Skip if not a directory
                if not os.path.isdir(protected_path):
                    continue

                # Print execution status
                print(f"Running analysis for {model_name}/{dataset_name}/{protected_name}")

                # Execute analysis for this combination
                try:
                    analysis(model_name, dataset_name, protected_name)
                except Exception as e:
                    print(f"Error in {model_name}/{dataset_name}/{protected_name}: {e}")


if __name__ == "__main__":
    analysis_all()
