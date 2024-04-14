import json
import os
import random
import numpy as np
import h5py
import skimage.io
from skimage.transform import resize


parent_path = '/Users/h0z058l/Downloads/FER/codes/Face-Recognition/BFW/3'

"""
-------- select 10 fold image paths ----------
"""
def select_images(path, num):
    all_images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):  # 检查文件类型，根据需要调整
                all_images.append(os.path.join(root, file))

    selected_images = random.sample(all_images, num)
    remaining_images = list(set(all_images) - set(selected_images))

    return selected_images, remaining_images


def create_image_to_label_map(path):
    selected_images, remaining_images = select_images(path, 250)

    for train_image in remaining_images:
        race_sex = train_image.split("/")[-3]
        train_x.append(train_image)
        train_y.append(race_sex)

    for test_image in selected_images:
        race_sex = test_image.split("/")[-3]
        test_x.append(test_image)
        test_y.append(race_sex)


for idx in range(10):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    img_parent_path = '/Users/h0z058l/Downloads/FER/dataset/BFW/bfw-cropped-aligned'
    race_sex_category = ["asian_females", "asian_males", "black_females", "black_males",
                         "indian_females", "indian_males", "white_females", "white_males"]
    for race_sex in race_sex_category:
        race_sex_path = os.path.join(img_parent_path, race_sex)
        create_image_to_label_map(race_sex_path)

    train_labels_set = set(train_y)
    train_labels_list = list(train_labels_set)
    train_index_to_label_map = {}
    train_label_to_index_map = {}
    for train_label_idx, label in enumerate(train_labels_list):
        train_index_to_label_map[train_label_idx] = label
        train_label_to_index_map[label] = train_label_idx

    test_labels_set = set(test_y)
    test_labels_list = list(test_labels_set)
    test_index_to_label_map = {}
    test_label_to_index_map = {}
    for test_label_idx, label in enumerate(test_labels_list):
        test_index_to_label_map[test_label_idx] = label
        test_label_to_index_map[label] = test_label_idx

    json_dict = {
        "train_path": train_x, "train_label": train_y,
        "test_path": test_x, "test_label": test_y,
        "train_idx2label_map": train_index_to_label_map,
        "train_label2idx_map": train_label_to_index_map,
        "test_idx2label_map": test_index_to_label_map,
        "test_label2idx_map": test_label_to_index_map
    }

    with open(f"{parent_path}/10-folder-selection/select_{idx}.json", 'w') as f:
        json.dump(json_dict, f)

print("selection finished")


"""
------------ create data file -------------
"""
for select_idx in range(10):
    train_pixels = []
    test_pixels = []

    with open(f"{parent_path}/10-folder-selection/select_{select_idx}.json") as f:
        selection = json.load(f)

    train_paths = selection["train_path"]
    train_labels = selection["train_label"]
    train_label_to_index_map = selection["train_label2idx_map"]
    test_paths = selection["test_path"]
    test_labels = selection["test_label"]
    test_label_to_index_map = selection["test_label2idx_map"]

    for train_path in train_paths:
        train_pixel = skimage.io.imread(train_path)
        resized_image = resize(train_pixel, (108, 124))
        train_pixels.append(resized_image.tolist())
    for test_path in test_paths:
        test_pixel = skimage.io.imread(test_path)
        resized_image = resize(test_pixel, (108, 124))
        test_pixels.append(resized_image.tolist())

    train_int_labels = []
    for train_label in train_labels:
        train_int_labels.append(train_label_to_index_map[train_label])
    test_int_labels = []
    for test_label in test_labels:
        test_int_labels.append(test_label_to_index_map[test_label])

    train_pixels = np.array(train_pixels)
    train_int_labels = np.array(train_int_labels)
    test_pixels = np.array(test_pixels)
    test_int_labels = np.array(test_int_labels)
    print(np.shape(train_pixels))
    print(np.shape(test_pixels))

    datafile = h5py.File(f'{parent_path}/data/data_{select_idx}.h5', 'w')
    datafile.create_dataset("Training_pixel", dtype='uint8', data=train_pixels)
    datafile.create_dataset("Training_label", dtype='int64', data=train_int_labels)
    datafile.create_dataset("test_pixels", dtype='uint8', data=test_pixels)
    datafile.create_dataset("Test_label", dtype='int64', data=test_int_labels)
    datafile.close()
    print(f"{select_idx} data files created")

print("Save data finish!!!")
