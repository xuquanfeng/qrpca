from setuptools import setup

with open('README.md', 'r',encoding='utf-8') as fp:
    long_description = fp.read()

setup(
    name='qrpca',
    version='1.1',
    description='A Python package for PCA algorithm of QR accelerated SVD decomposition with CUDA of torch.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/xuquanfeng/qrpca',
    keywords=['Astronomy data analysis', 'Astronomy toolbox', 'Dimensionality reduction'],
    author='Xu Quanfeng',
    author_email='xqf3520@163.com',
    packages=['qrpca'],
    classifiers=[
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.6",
)