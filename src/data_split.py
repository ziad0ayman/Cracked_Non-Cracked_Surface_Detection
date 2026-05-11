import os
import shutil
import json
import splitfolders

# 1. Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "..", "config", "config.json")
source_path = os.path.join(script_dir, "..", "data", "raw", "Bangunan Retak")
dest_path = os.path.join(script_dir, "..", "data", "split")

# 2. Read the configuration file
with open(config_path, "r") as file:
    config = json.load(file)
    
# Extract the ratios and convert the list to a tuple
dataset_ratios = tuple(config["split_ratios"])

# 3. Clean up the destination if it already exists
if os.path.exists(dest_path):
    shutil.rmtree(dest_path)

# 4. Split the dataset
print(f"Splitting data with ratios: {dataset_ratios}")
splitfolders.ratio(
    source_path, 
    output=dest_path, 
    seed=42,
    ratio=dataset_ratios, 
    group_prefix=None, 
    move=False
)
print(f"Dataset split into: {dest_path}")
