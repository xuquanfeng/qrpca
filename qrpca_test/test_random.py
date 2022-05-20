# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:22:29 2022

@author: xqf35
"""
import time
import numpy as np
from qrpca.decomposition import qrpca
from qrpca.decomposition import svdpca
import torch
np.set_printoptions(suppress=True)

X_train = torch.rand(60000,1000)
X_test = torch.rand(10000,1000)
n_com = 0.95

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
pca = qrpca(n_component_ratio=n_com,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
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
pca = qrpca(n_component_ratio=n_com,device=device)
X_train_qrpca = pca.fit_transform(X_train)
X_test_qrpca=pca.transform(X_test)
pca_n=X_train_qrpca.shape[1]
print('='*10,'torch_QR','='*10)
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
print(X_train_qrpca.device)
print("--- %s seconds ---" % (time.time() - start_time))

device = torch.device("cpu")
start_time = time.time()
pca = svdpca(n_component_ratio=n_com,device=device)
# pca = PCA(n_component_ratio=10) #When the parameter is an integer, n principal components are reserved.
X_train_qrpca = pca.fit_transform(X_train)
X_test_qrpca=pca.transform(X_test)
pca_n=X_train_qrpca.shape[1]
print('='*10,'torch_SVD','='*10)
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
print(X_train_qrpca.device)
print("--- %s seconds ---" % (time.time() - start_time))

from sklearn.decomposition import PCA

start_time = time.time()
pca2 = PCA(n_components=n_com, copy=True, whiten=False)
X_train_pca2 = pca2.fit_transform(X_train)
X_test_pca2 = pca2.transform(X_test)
pca_n2 = X_train_pca2.shape[1]
print('='*10,'sklearn','='*10)
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
print("--- %s seconds ---" % (time.time() - start_time))

