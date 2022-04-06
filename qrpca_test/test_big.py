# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:22:29 2022

@author: xqf35
"""
import time
import numpy as np
from qrpca.decomposition import QRPCA
from qrpca.decomposition import SVDPCA
import pandas as pd
import torch
np.set_printoptions(suppress=True)

# from sklearn.datasets import load_digits  # 经典手写数字分类数据集
# from sklearn.model_selection import train_test_split
# digits = load_digits()
# digits.target = pd.DataFrame(digits.target)
# digits.data = pd.DataFrame(digits.data)
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=1)

import torchvision
dataset_train = torchvision.datasets.MNIST(root="./",train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataset_test = torchvision.datasets.MNIST(root="./",train=False, transform=torchvision.transforms.ToTensor(), download=False)
X_train = dataset_train.train_data.numpy()
X_test = dataset_test.test_data.numpy()
X_train = pd.DataFrame(X_train.reshape(X_train.shape[0],-1))
X_test = pd.DataFrame(X_test.reshape(X_test.shape[0],-1))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
pca = QRPCA(n_component_ratio=0.95,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
# pca = PCA(n_component_ratio=10) #When the parameter is an integer, n principal components are reserved.
X_train_qrpca = pca.fit_transform(X_train)
X_test_qrpca=pca.transform(X_test)
pca_n=X_train_qrpca.shape[1]
print('='*10,'torch_QR','='*10)
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
print(X_train_qrpca.device)
print("--- %s seconds ---" % (time.time() - start_time))

device = torch.device("cpu")
start_time = time.time()
pca = QRPCA(n_component_ratio=0.95,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
# pca = PCA(n_component_ratio=10) #When the parameter is an integer, n principal components are reserved.
X_train_qrpca = pca.fit_transform(X_train)
X_test_qrpca=pca.transform(X_test)
pca_n=X_train_qrpca.shape[1]
print('='*10,'torch_QR','='*10)
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
print(X_train_qrpca.device)
print("--- %s seconds ---" % (time.time() - start_time))

from sklearn.decomposition import PCA

start_time = time.time()

pca2 = PCA(n_components=0.95, copy=True, whiten=False)
X_train_pca2 = pca2.fit_transform(X_train)
X_test_pca2 = pca2.transform(X_test)
pca_n2 = X_train_pca2.shape[1]
print('='*10,'sklearn','='*10)
print(
    "keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n2,
                                                                                                                 X_train_pca2.shape,
                                                                                                                 X_test_pca2.shape))
print("--- %s seconds ---" % (time.time() - start_time))

