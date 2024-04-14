import os
import json

# path = '/Users/h0z058l/Downloads/FER/dataset/BUPT-Globalface/images/race_unbalance/Indian'
# files = os.listdir(path)
# images_count = len([i for i in files if i.endswith(('.jpg', '.jpeg', '.png'))])
#
# print('图片数量：', images_count)

path = '/Users/h0z058l/Downloads/FER/dataset/BUPT-Balencedface/images/race_per_7000/Caucasian'
print(len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]))

# 种族下面是identity ID folder
# Asian: 7000
# African: 7000
# Caucasian: 7000
# Indian: 7000
