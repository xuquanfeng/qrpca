# QRPCA

A Python package for PCA algorithm of QR accelerated SVD decomposition with CUDA of torch.

`QRPCA` is a package that uses singular value decomposition and QR decomposition to perform PCA dimensionality reduction. It takes the two-dimensional matrix data matrix as the input, trains the PCA dimensionality reduction matrix, and reduces the dimension of the test data according to the training data. This method can accelerate the operation with GPU in torch environment. Consequently, this package can be used as a simple toolbox to perform astronomical data cleaning.

## How to install `QRPCA`
The `QRPCA` can be installed by the PyPI and pip:
```
pip install qrpca
```
If you download the repository, you can also install it in the `QRPCA` directory:
```
https://github.com/xuquanfeng/qrpca
```
You can access it by clicking on 
[Github-QRPCA](https://github.com/xuquanfeng/qrpca)
.
## How to Use `QRPCA`
Here is a demo for the use of `QRPCA`.
```commandline
import time
import numpy as np
import pandas as pd
import torch
from QRPCA import QRPCA
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
pca = QRPCA(n_component_ratio=0.95,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
# pca = PCA(n_component_ratio=10) #When the parameter is an integer, n principal components are reserved.
X_train_qrpca = pca.fit_transform(X_train)
X_test_qrpca=pca.transform(X_test)
pca_n=X_train_qrpca.shape[1]
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n,X_train_qrpca.shape,X_test_qrpca.shape))
print(X_train_qrpca.device)
print("--- %s seconds ---" % (time.time() - start_time))
```
Then the result is as follows:
```commandline
keep 28 features after pca 
the shape of X_train after PCA: torch.Size([1078, 28]) 
the shape of X_test after PCA: torch.Size([719, 28])
cuda:0
--- 2.390354633331299 seconds ---
```
## Requirements
- numpy>=1.21.1
- pandas>=1.3.5
- torch>=1.8.1
- cudatoolkit>=0.7.1
- scikit-learn>=1.0.2

Use the dependent environment as above, `scikit-python` is the dependent package required to load test data.

