QRPCA
=======

A Python package for PCA algorithm of QR accelerated SVD decomposition with CUDA of torch.

`QRPCA` is a package that uses singular value decomposition and QR decomposition to perform PCA dimensionality reduction. It takes the two-dimensional matrix data matrix as the input, trains the PCA dimensionality reduction matrix, and reduces the dimension of the test data according to the training data. This method can accelerate the operation with GPU in torch environment. Consequently, this package can be used as a simple toolbox to perform astronomical data cleaning.

## How to install `QRPCA`
The `QRPCA` can be installed by the PyPI and pip:
```
pip install qrpca
```
If you download the repository, you can also install it in the `QRPCA` directory:
```
https://github.com/xuquanfeng/qrpca/edit/qrpca
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
from qrpca import QRPCA
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
pca = QRPCA.QRPCA(n_component_ratio=0.95,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
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
Compare to the speed of the sklearn package:
```commandline
import time
import numpy as np
from qrpca import QRPCA
import pandas as pd
import torch
import torchvision

np.set_printoptions(suppress=True)
dataset_train = torchvision.datasets.MNIST(root="./",train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataset_test = torchvision.datasets.MNIST(root="./",train=False, transform=torchvision.transforms.ToTensor(), download=False)
X_train = dataset_train.train_data.numpy()
X_test = dataset_test.test_data.numpy()
X_train = pd.DataFrame(X_train.reshape(X_train.shape[0],-1))
X_test = pd.DataFrame(X_test.reshape(X_test.shape[0],-1))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start_time = time.time()
pca = QRPCA.QRPCA(n_component_ratio=0.95,device=device)  #When the parameter here is decimal, it is the percentage of information retained.
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
print("keep {} features after pca \nthe shape of X_train after PCA: {} \nthe shape of X_test after PCA: {}".format(pca_n2,
                                                                                                                 X_train_pca2.shape,
                                                                                                                 X_test_pca2.shape))
print("--- %s seconds ---" % (time.time() - start_time))
```
Then the result is as follows:
```commandline
========== torch_QR ==========
keep 154 features after pca 
the shape of X_train after PCA: torch.Size([60000, 154]) 
the shape of X_test after PCA: torch.Size([10000, 154])
cuda:0
--- 3.7280354499816895 seconds ---
========== sklearn ==========
keep 154 features after pca 
the shape of X_train after PCA: (60000, 154) 
the shape of X_test after PCA: (10000, 154)
--- 4.914863109588623 seconds ---
```
## Requirements
- numpy>=1.21.1
- pandas>=1.3.5
- torch>=1.8.1
- torchvision>=0.8.0
- cudatoolkit>=0.7.1
- scikit-learn>=1.0.2

Use the dependent environment as above, `scikit-python` is the dependent package required to load test data.
## Copyright & License
2022 Xu Quanfeng (xuquanfeng@shao.ac.cn) & Rafael S. de Souza (drsouza@shao.ac.cn)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
## References
- Sharma A, Paliwal K K, Imoto S, et al. Principal component analysis using QR decomposition[J]. International Journal of Machine Learning and Cybernetics, 2013, 4(6): 679-683.

## Citing ``qrpca``
If you want to cite ``qrpca``, please use the following citations.

Software Citation: Xu Quanfeng, & Rafael S. de Souza. (2022). PCA algorithm of QR accelerated SVD decomposition (1.1). Zenodo. https://doi.org/10.5281/zenodo.6362371
