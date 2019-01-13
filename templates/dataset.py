from torch.utils import data
import csv
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
# Workaround for RuntimeError: cuda runtime error (3).
# See https://discuss.pytorch.org/t/cuda-runtime-error-3-initialization-error-with-from-numpy-and-dataset/11827
mp.set_start_method('spawn')

class MLDataset(data.Dataset):

    def __init__(self, data):
        """
        TODO: read and process data here, and store as torch.tensors.
        Put entire dataset on GPU if possible.
        """
        self.inputs = None
        self.targets = None
        self.active_targets = self.targets
        self.active_inputs = self.inputs
        self.device = torch.device('cuda:0')

    def __getitem__(self, idx):
        return self.active_inputs[idx], self.active_targets[idx]


    def __len__(self):
        return len(self.active_targets)


    def set_active_data(self, indices):
        """
        Sets which training examples are active, so that one object can
        store all the training data but only return a subset of it for 
        cross-validation or training on part of the dataset.
        """
        self.active_targets = self.targets[indices]
        self.active_inputs = self.inputs[indices]
