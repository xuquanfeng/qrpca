qrpca
=====
.. image:: https://img.shields.io/pypi/v/qrpca
   :target: https://pypi.org/project/qrpca/

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6555926.svg
   :target: https://doi.org/10.5281/zenodo.6555926
.. image:: https://img.shields.io/badge/build-passing-successw

 *qrpca  works similarly to sklean.decomposition, but employs a QR-based PCA decomposition and supports CUDA acceleration via torch.*

`See documentation here! <https://qrpca.readthedocs.io/en/stable/README.html>`_

How to install ``qrpca``
========================

The ``qrpca`` can be installed by the PyPI and pip:

::

   pip install qrpca

If you download the repository, you can also install it in the ``qrpca`` directory:

::

   git clone https://github.com/xuquanfeng/qrpca
   cd qrpca
   python setup.py install

You can access it by clicking on `Github-qrpca <https://github.com/xuquanfeng/qrpca>`_

Usage
====================

Here is a demo for the use of `qrpca`.

The following are the results of retaining principal components containing 95% of the information content by principal component analysis.


You can set the parameter ``n_components`` to a value between 0 and 1 to execute the PCA on the corresponding proportion of the entire data, or set it to an integer number to reserve the ``n_omponents`` components.

::

    import torch
    import numpy as np
    from qrpca.decomposition import qrpca
    from qrpca.decomposition import svdpca
    
    # Generate the random data
    demo_data = torch.rand(60000,2000)
    n_com = 0.95

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # qrpca
    pca = qrpca(n_component_ratio=n_com,device=device) # The percentage of information retained.
    # pca = qrpca(n_component_ratio=10,device=device) # n principal components are reserved.
    demo_qrpca = pca.fit_transform(demo_data)
    print(demo_pca)
    
    # svdpca
    pca = svdpca(n_component_ratio=n_com,device=device)
    demo_svdpca = pca.fit_transform(demo_data)
    print(demo_svdpca)

==========================
Comparision with sklearn
==========================

The methods and usage of ``qrpca`` are almost identical to those of ``sklearn.decomposition.PCA``. If you want to switch from ``sklearn`` to ``qrpca``, all you have to do is change the import and declare the device if you have a GPU, and that's it.

And here's an illustration of how minimal the change is when different ``PCA`` is used:

- qrpca.decomposition.qrpca
::

    from qrpca.decomposition import qrpca
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pca = qrpca(n_component_ratio=n_com,device=device)
    demo_qrpca = pca.fit_transform(demo_data)

- qrpca.decomposition.svdpca
::

    from qrpca.decomposition import svdpca

    pca = svdpca(n_component_ratio=n_com)
    demo_svdpca = pca.fit_transform(demo_data)

- sklearn.decomposition.PCA
::

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_com)
    demo_pca = pca.fit_transform(demo_data)


=============================
Performance benchmark sklearn
=============================

With the acceleration of GPU computation, the speed of both QR decomposition and singular value decomposition in ``qrpca`` is much higher than that in ``sklearn``

We run the different PCA methods on data with different numbers of rows and columns, and then we compare their PCA degradation times and plotted the distribution of the times. Here are the two plots.

**Comparison of PCA degradation time with different number of rows and different methods for the case of 1000 columns.**

.. image:: https://github.com/xuquanfeng/qrpca/blob/v1.4.4/qrpca_test/result_1000.png

**Comparison of PCA reduction time with different number of columns and different methods for the case of 30000 rows.**

.. image:: https://github.com/xuquanfeng/qrpca/blob/v1.4.4/qrpca_test/3w_18_result.png


We can see from the above two facts that ``qrpca`` may considerably cut program run time by using GPU acceleration, while also having a very cheap migration cost and a guaranteed impact.

Requirements
============

-  numpy>=1.21.1
-  pandas>=1.3.5
-  torch>=1.8.1
-  torchvision>=0.8.0
-  cudatoolkit>=0.7.1
-  scikit-learn>=1.0.2

Copyright & License
===================
2022 Xu Quanfeng (xuquanfeng@shao.ac.cn) & Rafael S. de Souza (drsouza@shao.ac.cn) & Shen Shiyin (ssy@shao.ac.cn) & Peng Chen (pengchzn@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

References
==========

- Sharma, Alok and Paliwal, Kuldip K. and Imoto, Seiya and Miyano, Satoru 2013, International Journal of Machine Learning and Cybernetics, 4, 6, doi: `10.1007/s13042-012-0131-7 <https://link.springer.com/article/10.1007/s13042-012-0131-7>`_


Citing ``qrpca``
================

If you want to cite ``qrpca``, please use the following citations.

@article{souza_qrpca_2022,
	title = {qrpca: {A} package for fast principal component analysis with {GPU} acceleration},
	volume = {41},
	copyright = {CC0 1.0 Universal Public Domain Dedication},
	issn = {2213-1337},
	url = {https://www.sciencedirect.com/science/article/pii/S221313372200052X},
	doi = {https://doi.org/10.1016/j.ascom.2022.100633},
	abstract = {We present qrpca, a fast and scalable QR-decomposition principal component analysis package. The software, written in both R and python languages, makes use of torch for internal matrix computations, and enables GPU acceleration, when available. qrpca provides similar functionalities to prcomp (R) and sklearn (python) packages respectively. A benchmark test shows that qrpca can achieve computational speeds 10–20 × faster for large dimensional matrices than default implementations, and is at least twice as fast for a standard decomposition of spectral data cubes. The qrpca source code is made freely available to the community.},
	journal = {Astronomy and Computing},
	author = {Souza, R. S. de and Quanfeng, X. and Shen, S. and Peng, C. and Mu, Z.},
	year = {2022},
	keywords = {Astroinformatics, GPU computing, Principal component analysis},
	pages = {100633},
}

Or

Software Citation: Xu Quanfeng, & Rafael S. de Souza. (2022). PCA algorithm of QR accelerated SVD decomposition (1.5). Zenodo. https://doi.org/10.5281/zenodo.6555926
