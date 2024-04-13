from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class BFW(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File(
            '/Users/h0z058l/Downloads/FER/codes/bfw/data/data_0.h5', 'r', driver='core'
        )
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((18000, 108, 124, 3))

        elif self.split == 'Test':
            self.test_data = self.data['test_pixels']
            self.PublicTest_labels = self.data['Test_label']
            self.test_data = np.asarray(self.test_data)
            self.test_data = self.test_data.reshape((2000, 108, 124, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Test':
            img, target = self.test_data[index], self.PublicTest_labels[index]

        # 如果图像数据已经是彩色图像，确保它的形状是 (height, width, 3)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = Image.fromarray(img)
        else:
            # 否则，假设图像是灰度图像，将其转换为彩色图像
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Test':
            return len(self.test_data)
