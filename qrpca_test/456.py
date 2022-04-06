# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 20:10:44 2022

@author: xqf35
"""
import numpy as np
import pandas as pd
import glob
import fitz  # 导入本模块需安装PyMuPDF库
import os

# 将文件夹中所有jpg图片全部转换为一个指定名称的pdf文件，并保存至指定文件夹
def pic2pdf_1(file_dir, pdf_name):
    # img_path = file_dir
    # pdf_path = file_dir
    doc = fitz.open()
    img = file_dir
    imgdoc = fitz.open(img)
    pdfbytes = imgdoc.convertToPDF()
    imgpdf = fitz.open("pdf", pdfbytes)
    doc.insertPDF(imgpdf)
    doc.save(pdf_name)
    doc.close()
namee = "3w_18_result"
tt = pd.read_csv(namee+'.csv')
tim = np.array(tt)
lei = tim[:,0]
tim = tim[:,1:]
import matplotlib.pyplot as plt
plt.figure(facecolor='#FFFFFF', figsize=(8,5))
plt.plot(lei,tim[:,0], marker='o', label='Gpu_torch_QR', linewidth=1.0)
plt.plot(lei,tim[:,1], marker='v', label='Cpu_torch_QR', linewidth=1.0)
plt.plot(lei,tim[:,2], marker='*', label='Cpu_torch_SVD', linewidth=1.0)
plt.plot(lei,tim[:,3], marker='.', label='sklearn_SVD', linewidth=1.0)
plt.xlabel('Number of columns', fontsize=18)
# plt.xscale("log")
plt.ylabel('Running time (Seconds)', fontsize=18)
plt.yscale("log")
plt.legend()
plt.savefig(namee+'.png',dpi=300)
plt.show()
pic2pdf_1(namee+'.png', pdf_name=namee+'.pdf')