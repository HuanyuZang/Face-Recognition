import os
import json

# path = '/Users/h0z058l/Downloads/FER/dataset/BUPT-Globalface/images/race_unbalance/Indian'
# files = os.listdir(path)
# images_count = len([i for i in files if i.endswith(('.jpg', '.jpeg', '.png'))])
#
# print('图片数量：', images_count)

path = '/Users/h0z058l/Downloads/FER/dataset/BUPT-Globalface/images/race_unbalance/African'
print(len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]))

# 种族下面是identity ID folder
# Asian: 11749
# African: 5157
# Caucasian: 14735
# Indian: 7096
