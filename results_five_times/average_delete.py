import os
import glob

def delete_average_files(root_dir):
    # "**/*_average.txt" でサブフォルダも含めて検索
    pattern = os.path.join(root_dir, "**", "*_average.txt")
    files = glob.glob(pattern, recursive=True)

    count = 0
    for file_path in files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            count += 1
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    print(f"\nTotal deleted: {count} file(s).")


# ---- 実行例 ----
delete_average_files("./")  # resultsフォルダ内の平均ファイルを削除
