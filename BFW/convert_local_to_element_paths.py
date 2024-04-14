import json

local_path = "/Users/h0z058l/Downloads/FER/codes/Face-Recognition/BFW/3/10-folder-selection"

for i in range(10):
    with open(f"{local_path}/select_{i}.json") as f:
        data = json.load(f)
    train_paths = data["train_path"]
    new_train_paths = [
        path.replace("/Users/h0z058l/Downloads/FER/dataset", "/shared/huanyu")
        for path in train_paths
    ]
    data["train_path"] = new_train_paths
    train_labels = data["train_label"]
    test_paths = data["test_path"]
    new_test_paths = [
        path.replace("/Users/h0z058l/Downloads/FER/dataset", "/shared/huanyu")
        for path in test_paths
    ]
    data["test_path"] = new_test_paths
    test_labels = data["test_label"]

    with open(f"{local_path}/select_{i}.json", "w") as f:
        json.dump(data, f)
