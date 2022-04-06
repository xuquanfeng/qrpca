# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:49:38 2022

@author: xqf35
"""
import time
import numpy as np
import pandas as pd
import torch
import time
np.set_printoptions(suppress=True)

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        AttributeError(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum",
        )
    return out

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
    # def __init__(self,n_component_ratio):
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
        explained_variance_ratio = explained_variance / total_var
        # explained_variance_ratio = list(explained_variance_ratio)
        if self.n_component_ratio>0 and self.n_component_ratio<1:
            ratio_cumsum = stable_cumsum(explained_variance_ratio.cpu().numpy())
            self.n_components = np.searchsorted(ratio_cumsum, self.n_component_ratio, side="right") + 1
            # self.n_components=self.__choose_ratio(explained_r=explained_variance_ratio)  #find how many features we should keep on the compressed rate we select
        elif type(self.n_component_ratio)==int and self.n_component_ratio<n_features:
            self.n_components = self.n_component_ratio
        else:
            AttributeError("n_components error!")
        x_compressed = torch.matmul(U[:, :self.n_components],torch.diag(s[:self.n_components]))
        bb = torch.matmul(q,x_compressed)
        del X_center,explained_variance,U,s,Vt,q      #del variable and release memory
        self.explained_variance_ratio=explained_variance_ratio[:self.n_components]
        return bb

    def transform(self,y):
        y = np.array(y)
        y = torch.tensor(y,dtype=torch.float32).to(self.device)
        y_centre = y-torch.mean(y, axis=0)
        q ,r = torch.linalg.qr(y_centre)
        y_compressed=torch.matmul(r,torch.linalg.inv(self.__Vt)[:,:self.n_components])
        del y_centre,r,y  #del variable and release memory
        # gc.collect()
        bb = torch.matmul(q,y_compressed)
        return bb

class SVDPCA(object):
    def __init__(self,n_component_ratio,device):
        self.n_component_ratio=n_component_ratio
        self.device = device

    def fit_transform(self, x):
        x = np.array(x)
        x = torch.tensor(x,dtype=torch.float32).to(self.device)
        n_samples, n_features = x.shape
        X_center = x-torch.mean(x, axis=0)   # Center data
        U, s, Vt = torch.linalg.svd(X_center, full_matrices=False)
        self.__Vt = Vt
        explained_variance = (s ** 2) / (n_samples - 1)  # Get variance explained by singular values
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var  # caculate the compressed rate
        if self.n_component_ratio > 0 and self.n_component_ratio < 1:
            ratio_cumsum = stable_cumsum(explained_variance_ratio.cpu().numpy())
            self.n_components = np.searchsorted(ratio_cumsum, self.n_component_ratio, side="right") + 1
        elif type(self.n_component_ratio) == int and self.n_component_ratio < n_features:
            self.n_components = self.n_component_ratio
        x_compressed = torch.matmul(U[:, :self.n_components], torch.diag(s[:self.n_components]))
        del X_center, explained_variance, U, s, Vt  # del variable and release memory
        self.explained_variance_ratio = explained_variance_ratio[:self.n_components]  # make explained variance ratio the attributes in pca, so that we can print it out
        return x_compressed

    def transform(self, y):
        y = np.array(y)
        y = torch.tensor(y,dtype=torch.float32).to(self.device)
        y_centre = y-torch.mean(y, axis=0)
        y_compressed=torch.matmul(y_centre,torch.linalg.inv(self.__Vt)[:,:self.n_components])
        del y_centre  # del variable and release memory
        return y_compressed