import os
import shutil
import json
import kagglehub
import splitfolders

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "..", "..", "config", "config.json")
raw_dir = os.path.join(script_dir, "..", "..", "data", "raw")
split_dir = os.path.join(script_dir, "..", "..", "data", "split")


def find_raw_dataset(root):
    """Walk raw_dir and return the first folder that contains Cracked/ and Non Cracked/."""
    for dirpath, dirnames, _ in os.walk(root):
        folders = {d.strip().lower() for d in dirnames}
        if "cracked" in folders and "non cracked" in folders:
            return dirpath
    return None


def download_dataset():
    with open(config_path, "r") as f:
        config = json.load(f)
    dataset_name = config["dataset_link"]

    source_path = kagglehub.dataset_download(dataset_name)
    print(f"Downloaded to cache: {source_path}")

    contents = os.listdir(source_path)
    if len(contents) == 1:
        single = os.path.join(source_path, contents[0])
        if os.path.isdir(single):
            source_path = single

    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    shutil.move(source_path, raw_dir)
    print(f"Moved to: {raw_dir}")


def split_dataset(source):
    with open(config_path, "r") as f:
        config = json.load(f)
    ratios = tuple(config["split_ratios"])

    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    print(f"Splitting with ratios {ratios} ...")
    splitfolders.ratio(source, output=split_dir, seed=42,
                       ratio=ratios, group_prefix=None, move=False)
    print(f"Done → {split_dir}/{{train,val,test}}")


def main():
    # 1. Already split?
    if os.path.isdir(split_dir):
        print(f"✓ Split data exists at '{split_dir}' — nothing to do.")
        return

    # 2. Raw data present?
    raw_source = find_raw_dataset(raw_dir)
    if raw_source is not None:
        print(f"✓ Raw data found at '{raw_source}' — splitting...")
        split_dataset(raw_source)
        return

    # 3. Neither — download then split
    print("No data found. Downloading from Kaggle...")
    download_dataset()
    raw_source = find_raw_dataset(raw_dir)
    if raw_source is None:
        print("ERROR: Could not locate Cracked/Non Cracked folders after download.")
        return
    split_dataset(raw_source)


if __name__ == "__main__":
    main()
