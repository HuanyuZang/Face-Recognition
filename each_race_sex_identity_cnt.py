import os
import json

def count_folders(path):
    return len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

path = '/Users/h0z058l/Downloads/FER/dataset/BFW/bfw-cropped-aligned/indian_males'
print(count_folders(path))

for i in range(10):
    with open(f'/Users/h0z058l/Downloads/FER/codes/bfw/10-folder-selection/select_{i}.json') as f:
        data = json.load(f)
    train_labels = set(data['train_label'])
    test_labels = set(data['test_label'])
    print(f'number {i} has {len(train_labels)} train_labels and {len(test_labels)} test_labels')
