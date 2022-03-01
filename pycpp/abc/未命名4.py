# -*- coding: utf-8 -*-
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
traindfy = pd.read_excel('Molecular_Descriptor.xlsx',sheet_name="training")
traindfy.head()
train_x = traindf.iloc[:,1:]
train_y = traindfy.iloc[:,2]
train_x.head()
train_y.plot(kind = "hist",bins = 50,figsize = (10,6))
plt.show()
train_xs = StandardScaler().fit_transform(train_x)
train_xs[0:5,:]
train_xname = train_x.columns.values
plt.figure(figsize=(20,20))

for ii in np.arange(49):
    plt.subplot(7,7,ii+1)
    plt.scatter(train_xs[:,ii],train_y)
    plt.title(train_xname[ii])
plt.tight_layout()
plt.figure()
plt.show()
varcorr = []
for ii in range(len(train_xname)):
    corrii = np.corrcoef(train_xs[:,ii],train_y.values)
    varcorr.append(corrii[0,1])
varcorrdf = pd.DataFrame({"varname":train_xname,"mycorr":varcorr})
varcorrdf.head()
varcorrdf.isna().sum()
varcorrdf.varname[varcorrdf.mycorr.isna()]
traindf[varcorrdf.varname[varcorrdf.mycorr.isna()]].apply(np.unique) 
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
from sklearn.feature_selection import SelectKBest, mutual_info_regression
## 通过方差分析的F值选择K个变量
KbestF = SelectKBest(mutual_info_regression, k=50)
KbestF_train_xs = KbestF.fit_transform(train_xs,train_y.values)
print(KbestF_train_xs.shape)
train_xname[KbestF.get_support()]

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

## 利用随机森林模型进行特征的选择
rfc = RandomForestRegressor(n_estimators=100,random_state=0)
rfc = rfc.fit(train_xs,train_y.values) # 使用模型拟合数据
## 定义从模型中进行特征选择的选择器
sfm = SelectFromModel(estimator=rfc, ## 进行特征选择的模型
                      prefit = True, ## 对模型进行预训练
                      max_features = 30,##选择的最大特征数量
                     )
## 将模型选择器作用于数据特征
sfm_train_x = sfm.transform(train_xs)
print(sfm_train_x.shape)
train_xname[sfm.get_support()]
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
## 将数据切分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(
    train_xs,train_y.values,test_size = 0.25,random_state = 1)
print("X_train.shape :",X_train.shape)
print("X_test.shape :",X_test.shape)
rfr1 = RandomForestRegressor(n_estimators=100,random_state = 1)
rfr1 = rfr1.fit(X_train,y_train)
## 计算在训练和和测试集上的预测均方根误差
rfr1_lab = rfr1.predict(X_train)
rfr1_pre = rfr1.predict(X_test)
print("训练数据集上的均方根误差:",mean_squared_error(y_train,rfr1_lab))
print("测试数据集上的均方根误差:",mean_squared_error(y_test,rfr1_pre))
print("训练数据集上的平均绝对误差:",mean_absolute_error(y_train,rfr1_lab))
print("测试数据集上的平均绝对误差:",mean_absolute_error(y_test,rfr1_pre))
plt.figure(1)
plt.figure(figsize=(16,8))
plt.subplot(1,2,1) ## 训练数据结果可视化
rmse = round(mean_squared_error(y_train,rfr1_lab),4)
index = np.argsort(y_train)

plt.plot(np.arange(len(index)),y_train[index],"r",
         linewidth=2, label = "原始数据")
plt.plot(np.arange(len(index)),rfr1_lab[index],"bo",
         markersize=3,label = "预测值")
plt.text(200,8,s = "均方根误差:"+str(rmse))
plt.legend()
plt.grid()
plt.xlabel("Index")
plt.ylabel("Y")
plt.title("随机森林回归(训练集)")

plt.subplot(1,2,2)   ## 测试数据结果可视化
rmse = round(mean_squared_error(y_test,rfr1_pre),4)
index = np.argsort(y_test)
plt.plot(np.arange(len(index)),y_test[index],"r",
         linewidth=2, label = "原始数据")
plt.plot(np.arange(len(index)),rfr1_pre[index],"bo",
         markersize=3,label = "预测值")
plt.text(50,8,s = "平均绝对误差:"+str(rmse))
plt.legend()
plt.grid()
plt.xlabel("Index")
plt.ylabel("Y")
plt.title("随机森林回归(测试集)")
plt.tight_layout()
plt.show()
importances = pd.DataFrame({"feature":train_xname,
                            "importance":rfr1.feature_importances_})
importances = importances.sort_values("importance",ascending = True)
importances.iloc[0:40,:].plot(kind="barh",figsize=(10,10),x = "feature",y = "importance",
                 legend = False)
plt.xlabel("重要性得分")
plt.ylabel("")
plt.title("随机森林回归")
plt.grid()
plt.show()