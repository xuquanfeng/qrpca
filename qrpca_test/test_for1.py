# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:22:29 2022

@author: xqf35
"""
import time
import numpy as np
from qrpca import QRPCA
import torch
np.set_printoptions(suppress=True)

leii = list(range(11))
leii[0] = 1
lei = []
tt = []
for j in leii:
    t = []
    # if i<j:
    #     break
    a = 10**4
    b = j*1000
    lei.append(b)
    print(a,b)
    X_train = torch.rand(a,b)
    X_test = torch.rand(int(a/10),b)
    n_com = 0.95

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    pca = QRPCA.QRPCA(n_component_ratio=n_com,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
    # pca = PCA(n_component_ratio=10) #When the parameter is an integer, n principal components are reserved.
    X_train_qrpca = pca.fit_transform(X_train)
    X_test_qrpca=pca.transform(X_test)
    pca_n=X_train_qrpca.shape[1]
    print('='*10,'torch_QR','='*10)
    print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
    print(X_train_qrpca.device)
    print("--- %s seconds ---" % (time.time() - start_time))
    t.append(time.time() - start_time)

    device = torch.device("cpu")
    start_time = time.time()
    pca = QRPCA.QRPCA(n_component_ratio=n_com,device=device)
    X_train_qrpca = pca.fit_transform(X_train)
    X_test_qrpca=pca.transform(X_test)
    pca_n=X_train_qrpca.shape[1]
    print('='*10,'torch_QR','='*10)
    print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
    print(X_train_qrpca.device)
    print("--- %s seconds ---" % (time.time() - start_time))
    t.append(time.time() - start_time)

    device = torch.device("cpu")
    start_time = time.time()
    pca = QRPCA.SVDPCA(n_component_ratio=n_com,device=device)
    # pca = PCA(n_component_ratio=10) #When the parameter is an integer, n principal components are reserved.
    X_train_qrpca = pca.fit_transform(X_train)
    X_test_qrpca=pca.transform(X_test)
    pca_n=X_train_qrpca.shape[1]
    print('='*10,'torch_SVD','='*10)
    print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
    print(X_train_qrpca.device)
    print("--- %s seconds ---" % (time.time() - start_time))
    t.append(time.time() - start_time)

    from sklearn.decomposition import PCA

    start_time = time.time()
    pca2 = PCA(n_components=n_com, copy=True, whiten=False)
    X_train_pca2 = pca2.fit_transform(X_train)
    X_test_pca2 = pca2.transform(X_test)
    pca_n2 = X_train_pca2.shape[1]
    print('='*10,'sklearn','='*10)
    print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
    print("--- %s seconds ---" % (time.time() - start_time))
    t.append(time.time() - start_time)
    tt.append(t)
print(tt)
tim = np.array(tt)
lei = np.array(lei)[1:]
import matplotlib.pyplot as plt
plt.figure(facecolor='#FFFFFF', figsize=(8,5))
plt.plot(lei,tim[1:,0], marker='o', label='Gpu_torch_QR', linewidth=1.0)
plt.plot(lei,tim[1:,1], marker='v', label='Cpu_torch_QR', linewidth=1.0)
plt.plot(lei,tim[1:,2], marker='*', label='Cpu_torch_SVD', linewidth=1.0)
plt.plot(lei,tim[1:,3], marker='.', label='skearn_SVD', linewidth=1.0)
plt.xlabel('Number of columns', fontsize=18)
# plt.xscale("log")
plt.ylabel('Running time (Seconds)', fontsize=18)
# plt.yscale("log")
plt.legend()
plt.savefig('result3.png',dpi=300)
plt.show()

