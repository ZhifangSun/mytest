#config InlineBackend.figure_format = 'retina'
#matplotlib inline
import seaborn as sns
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False # 解决坐标轴的负号显示问题
## 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import *
from sklearn.tree import *# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:54:42 2021

@author: 55394
"""

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline
import seaborn as sns
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False # 解决坐标轴的负号显示问题
## 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import *
from sklearn.tree import *

## 数据导入
traindf = pd.read_excel('Molecular_Descriptor.xlsx',sheet_name="training")
traindf.head()
traindfy = pd.read_excel('ER_activity.xlsx',sheet_name="training")
traindfy.head()
#数据切片
train_x = traindf.iloc[:,1:]
train_y = traindfy.iloc[:,2]
train_x.head()
#train_y.plot(kind = "hist",bins = 50,figsize = (10,6))
#plt.show()
#训练数据的标准化处理
train_xs = StandardScaler().fit_transform(train_x)
train_xs[0:5,:]
train_xname = train_x.columns.values
plt.figure(figsize=(20,20))

#for ii in np.arange(49):
#    plt.subplot(7,7,ii+1)
#   plt.scatter(train_xs[:,ii],train_y)
#    plt.title(train_xname[ii])
#plt.tight_layout()
#plt.figure()
#plt.show()
varcorr = []
for ii in range(len(train_xname)):
    corrii = np.corrcoef(train_xs[:,ii],train_y.values)
    varcorr.append(corrii[0,1])
varcorrdf = pd.DataFrame({"varname":train_xname,"mycorr":varcorr})
varcorrdf.head()
varcorrdf.isna().sum()
varcorrdf.varname[varcorrdf.mycorr.isna()]
traindf[varcorrdf.varname[varcorrdf.mycorr.isna()]].apply(np.unique)
a=varcorr.sort()
print(a)
## 提出不重要的225个变量
index = varcorrdf.varname[~varcorrdf.mycorr.isna()].index.values
train_xs = train_xs[:,index]
train_xname = train_xname[index]
train_xname[1:5]
varcorrdf.mycorr.plot(kind = "hist",bins = 30,figsize = (12,7))
plt.show()

## 基于统计方法选择

## （1）删除低方差的特征
from sklearn.feature_selection import VarianceThreshold
VTH = VarianceThreshold(threshold = 1)
train_xs_new = VTH.fit_transform(train_xs)
print(train_xs_new.shape)
from sklearn.feature_selection import SelectKBest, mutu
## 数据导入
traindf = pd.read_excel("Molecular_Descriptor.xlsx",sheet_name="training")
traindf.head()
## 导入数据的待预测值
traindfy = pd.read_excel("ER_activity.xlsx",sheet_name="training")
traindfy.head()
train_x = traindf.iloc[:,1:]
train_y = traindfy.iloc[:,2]
train_x.head()
train_y.plot(kind = "hist",bins = 50,figsize = (10,6))
plt.show()
## 训练数据的标准化处理
train_xs = StandardScaler().fit_transform(train_x)
train_xs[0:5,:]
## 可视化其中的几个变量与因变量Y的散点图，看一下数据的分布
train_xname = train_x.columns.values
plt.figure(figsize=(20,20))

for ii in np.arange(64):
    plt.subplot(8,8,ii+1)
    plt.scatter(train_xs[:,ii],train_y)
    plt.title(train_xname[ii])
plt.tight_layout()
plt.show()

## 数据的分布情况各种各样链接：https://pan.baidu.com/s/1kC7oPBOP7yq4BQHCJ0pqmw
#提取码：ov41
