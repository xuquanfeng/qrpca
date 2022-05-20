# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:49:38 2022

@author: xqf35
"""
import time
import numpy as np
import pandas as pd
import torch
from qrpca.decomposition import qrpca
from qrpca.decomposition import svdpca
np.set_printoptions(suppress=True)

from sklearn.datasets import load_digits # 经典手写数字分类数据集
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
digits= load_digits()
digits.target=pd.DataFrame(digits.target)
digits.data=pd.DataFrame(digits.data)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=1)

start_time = time.time()
pca = qrpca(n_component_ratio=0.95,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
# pca = PCA(n_component_ratio=10) #When the parameter is an integer, n principal components are reserved.
X_train_qrpca = pca.fit_transform(X_train)
X_test_qrpca=pca.transform(X_test)
pca_n=X_train_qrpca.shape[1]
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
print(X_train_qrpca.device)
print("--- %s seconds ---" % (time.time() - start_time))

'''
### For test sklearn package
from sklearn.decomposition import PCA
start_time = time.time()

pca2 = PCA(n_components=0.95,copy=True, whiten=False)
X_train_pca2 = pca2.fit_transform(X_train)
X_test_pca2=pca2.transform(X_test)
pca_n2=X_train_pca2.shape[1]
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n2,X_train_pca2.shape,X_test_pca2.shape))

print("--- %s seconds ---" % (time.time() - start_time))


'''