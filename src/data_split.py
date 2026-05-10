import os
import shutil
import splitfolders

# 1. Get the absolute path to the folder containing this exact .py script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Build the destination path relative to the script
dest_path = os.path.join(script_dir, "..", "data", "split")

# If data/split already exists from a previous run, delete it first
if os.path.exists(dest_path):
    shutil.rmtree(dest_path)

# 3. Get the source path for the raw data
source_path = os.path.join(script_dir, "..", "data", "raw")

# 4. Split the dataset into train, validation, and test sets
splitfolders.ratio(source_path, output=dest_path, ratio=(.8, .15, .05), group_prefix=None, move=False)