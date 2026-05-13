import os
import shutil
import random
from pathlib import Path
import kagglehub

from src.utils.config import get_dataset_link, get_split_ratios

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, "..", "..", "data", "raw")
split_dir = os.path.join(script_dir, "..", "..", "data", "split")


def find_raw_dataset(root):
    for dirpath, dirnames, _ in os.walk(root):
        folders = {d.strip().lower() for d in dirnames}
        if "cracked" in folders and "non cracked" in folders:
            return dirpath
    return None


def download_dataset():
    dataset_name = get_dataset_link()

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


def split_dataset(source, seed=42):
    ratios = get_split_ratios()

    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    split_names = ["train", "val", "test"]
    print(f"Splitting with ratios {ratios} ...")

    source = Path(source)
    classes = [d for d in source.iterdir() if d.is_dir()]

    for cls in classes:
        files = sorted(cls.iterdir())
        rng = random.Random(seed)
        rng.shuffle(files)

        n_total = len(files)
        n1 = int(ratios[0] * n_total)
        n2 = int(ratios[1] * n_total)
        splits = [files[:n1], files[n1:n1 + n2], files[n1 + n2:]]

        for split_name, split_files in zip(split_names, splits):
            out_dir = Path(split_dir) / split_name / cls.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(str(f), str(out_dir / f.name))

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
