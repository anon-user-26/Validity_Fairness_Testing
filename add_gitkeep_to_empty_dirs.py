import os

def add_gitkeep_to_empty_dirs(root_dir):
    for current_dir, dirs, files in os.walk(root_dir):
        if not files and not dirs:
            gitkeep_path = os.path.join(current_dir, '.gitkeep')
            f = open(gitkeep_path, 'w')
            f.close()
            print("Added: " + gitkeep_path)

if __name__ == "__main__":
    add_gitkeep_to_empty_dirs(".")