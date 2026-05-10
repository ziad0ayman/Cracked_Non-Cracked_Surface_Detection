import os
import shutil
import json
import kagglehub

# 1. Anchor the destination path to wherever this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
dest_path = os.path.join(script_dir, "..", "data", "raw")
config_path = os.path.join(script_dir, "..", "config", "config.json") # Path to your new config file

# Read the config file
with open(config_path, 'r') as f:
    config = json.load(f)
dataset_name = config["dataset_link"]

# 2. Download the dataset using kagglehub
source_path = kagglehub.dataset_download(dataset_name)
print(f"Dataset temporarily downloaded to cache: {source_path}")

# 3. Check for nested root folders
contents = os.listdir(source_path)

# If the download folder contains exactly ONE item, and it's a directory...
if len(contents) == 1:
    single_item_path = os.path.join(source_path, contents[0])
    if os.path.isdir(single_item_path):
        # ...update  source path to dive inside that inner folder
        source_path = single_item_path

# 4. Clean up the destination if it already exists from a previous run
if os.path.exists(dest_path):
    shutil.rmtree(dest_path)

# 5. Move the dataset
shutil.move(source_path, dest_path)

print(f"Dataset moved to: {dest_path}")
