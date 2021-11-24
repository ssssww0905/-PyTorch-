# coding=UTF-8
"""
0. Split dataset
    "cat.0.jpg" - "cat.12499.jpg"
    "dog.0.jpg" - "dog.12499.jpg"

    import os
    import shutil
"""
import os  # os.path
from shutil import copyfile, rmtree
import random


def main():
    random.seed(1)
    origin_path = "./dog_cat_train"
    train_path = "./train"
    val_path = "./val"

    assert os.path.exists(origin_path)
    if os.path.exists(train_path):
        rmtree(train_path)
    if os.path.exists(val_path):
        rmtree(val_path)


    val_rate = 0.1
    dog_val_index_list = random.sample(range(12500), int(12500 * val_rate))
    cat_val_index_list = random.sample(range(12500), int(12500 * val_rate))
    val_index_dict = {"dog": dog_val_index_list, "cat": cat_val_index_list}

    print("split start!")
    image_names = os.listdir(origin_path)
    for image_name in image_names:
        target, index, _ = image_name.split(".")
        src_path = os.path.join(origin_path, image_name)
        if int(index) in val_index_dict[target]:
            dst_path = os.path.join(val_path, target)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            copyfile(src_path, os.path.join(dst_path, image_name))
        else:
            dst_path = os.path.join(train_path, target)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            copyfile(src_path, os.path.join(dst_path, image_name))

    print("split done!")


if __name__ == "__main__":
    main()
