# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:49:38 2022

@author: xqf35
"""
import time
import numpy as np
import pandas as pd
import torch
np.set_printoptions(suppress=True)
class QRPCA(object):
    """
    Principal component analysis (PCA).
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.
    Parameters
    ----------
    n_component_ratio: select the number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_component_ratio
    ----------
    
    function1-fit_transform: Fit the model by computing full SVD on X.
    Parameters
    ----------
    X: x_train
    ----------
    
    function2-transform: Fit the model with X and apply the dimensionality reduction on y.
    Parameters
    ----------
    y : x_test
    ----------
    """
    
    def __init__(self,n_component_ratio,device):
        self.n_component_ratio=n_component_ratio
        self.device = device
        
    def fit_transform(self,x):
        x = np.array(x)
        x = torch.tensor(x,dtype=torch.float32).to(self.device)
        n_samples, n_features = x.shape       
        X_center = x-torch.mean(x, axis=0)   # Center data
        q ,r = torch.linalg.qr(X_center)
        U, s, Vt=torch.linalg.svd(r, full_matrices=False)
        # self.q = q
        self.__Vt=Vt
        explained_variance = (s ** 2) / (n_samples - 1)   # Get variance explained by singular values
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var   #caculate the compressed rate
        if self.n_component_ratio>0 and self.n_component_ratio<1:
            self.n_components=self.__choose_ratio(explained_r=explained_variance_ratio)  #find how many features we should keep on the compressed rate we select
        elif type(self.n_component_ratio)==int and self.n_component_ratio<n_features:
            self.n_components=self.n_component_ratio
        x_compressed = torch.matmul(U[:, :self.n_components],torch.diag(s[:self.n_components]))  #return the features we choose
        del X_center,explained_variance,U,s,Vt      #del variable and release memory
        self.explained_variance_ratio=explained_variance_ratio[:self.n_components] #make explained variance ratio the attributes in pca, so that we can print it out
        return torch.matmul(q,x_compressed)
    
    def __choose_ratio(self,explained_r):
        for i in range(1, len(explained_r)):
            if sum(explained_r[:i])/sum(explained_r) >= self.n_component_ratio:
                return i
            
    def transform(self,y):
        y = np.array(y)
        y = torch.tensor(y,dtype=torch.float32).to(self.device)
        y_centre = y-torch.mean(y, axis=0)
        q ,r = torch.linalg.qr(y_centre)
        y_compressed=torch.matmul(r,torch.linalg.inv(self.__Vt)[:,:self.n_components])   # compress x_test based on the Vt on x_train
        del y_centre,r  #del variable and release memory
        # gc.collect()
        bb = torch.matmul(q,y_compressed)
        return bb
if __name__ == '__main__':
    from sklearn.datasets import load_digits # 经典手写数字分类数据集
    from sklearn.model_selection import train_test_split
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digits= load_digits()
    digits.target=pd.DataFrame(digits.target)
    digits.data=pd.DataFrame(digits.data)
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=1)
    
    start_time = time.time()
    pca = QRPCA(n_component_ratio=0.95,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
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