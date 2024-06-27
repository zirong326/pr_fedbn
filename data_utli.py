import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class DigitsDataset(Dataset):
    def __init__(self, data_path, channels, num_partitions=2, filename=None, train=True):
        if filename is None:
            if train:
                if num_partitions >1:
                    for part in range(int(num_partitions)):
                        if part == 0:
                            self.datas, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            datas, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.datas = np.concatenate([self.datas,datas], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.datas, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.datas.shape[0] /num_partitions)
                    self.datas = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.datas, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.datas, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)


    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, idx):
        feature = self.datas[idx]
        label = self.labels[idx]
        return image, label
