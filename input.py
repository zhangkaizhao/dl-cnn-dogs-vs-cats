import os
import os.path
import random
import shutil

from tqdm import tqdm

from settings import TRAIN_RATIO, TRAIN_TOTAL


def prepare_directories():
    # clear old training data
    if os.path.isdir("data"):
        shutil.rmtree("data")

    # new training data
    os.makedirs("data/train/cats")
    os.makedirs("data/train/dogs")
    os.makedirs("data/validation/cats")
    os.makedirs("data/validation/dogs")

    # for model savings
    if not os.path.isdir("models"):
        os.makedirs("models")

    # for features savings
    if not os.path.isdir("features"):
        os.makedirs("features")


# prepare datasets


def organize_datasets(
        train_data_dirpath,
        train_total=TRAIN_TOTAL,
        train_ratio=TRAIN_RATIO):
    all_files = [
        os.path.join(train_data_dirpath, p)
        for p in os.listdir(train_data_dirpath)
    ]
    train_files = random.sample(all_files, train_total)
    train_number = int(train_total * train_ratio)
    to_val, to_train = train_files[:train_number], train_files[train_number:]

    cats_train_dirpath = os.path.join("data", "train", "cats")
    dogs_train_dirpath = os.path.join("data", "train", "dogs")

    for t in tqdm(to_train):
        if "cat" in t:
            shutil.copy2(t, cats_train_dirpath)
        else:
            shutil.copy2(t, dogs_train_dirpath)

    cats_val_dirpath = os.path.join("data", "validation", "cats")
    dogs_val_dirpath = os.path.join("data", "validation", "dogs")
    for v in tqdm(to_val):
        if "cat" in v:
            shutil.copy2(v, cats_val_dirpath)
        else:
            shutil.copy2(v, dogs_val_dirpath)


if __name__ == "__main__":
    prepare_directories()
    organize_datasets("all/train")
