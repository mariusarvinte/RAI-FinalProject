# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:14:18 2019

@author: ma56473
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# load normalized COIL-20 dataset
def load_coil20(folder, seed=8):
    x_train = []
    y_train = []
    file_list = os.listdir(folder)
    # Shuffle in place
    np.random.seed(seed)
    np.random.shuffle(file_list)
    for filename in file_list:
        img = plt.imread(folder+'/'+filename)
        x_train.append(img)
        filename_split = filename.split("__")
        filename_split = filename_split[0]
        filename_split2 = filename_split.split("obj")
        y_train.append(int(filename_split2[1]))
    return x_train, y_train