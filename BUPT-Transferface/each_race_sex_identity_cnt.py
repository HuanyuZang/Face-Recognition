import os
import json

path = '/Users/h0z058l/Downloads/FER/dataset/BUPT-Transferface/train/data/Indian'
files = os.listdir(path)
images_count = len([i for i in files if i.endswith(('.jpg', '.jpeg', '.png'))])

print('图片数量：', images_count)

path = '/Users/h0z058l/Downloads/FER/dataset/BUPT-Transferface/train/data/Caucasian'
print(len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]))

# Caucasian底下是identity ID folder，然后每个folder底下是图
# 除了Caucasian, 其他种族下面都是直接图
# Asian: 54188
# African: 50588
# Caucasian: 10000
# Indian: 52285
