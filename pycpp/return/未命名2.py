# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:33:10 2021

@author: 55394
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


# random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn import svm
from sklearn.svm import SVR
# define dataset
dataset= pd.read_excel('Molecular_Descriptor.xlsx')


targetset= pd.read_excel('ERα_activity.xlsx')

data=pd.concat([dataset.iloc[:,1:],targetset.iloc[:,2]],axis=1)
NP = preprocessing.MaxAbsScaler()
#归一化到[-1，1]
data = NP.fit_transform(data)
x= data[:,:729]
y= data[:,729]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# define the model

model = RandomForestRegressor()
# fit the model
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
model.score(x_test,y_test)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.figure(1)
pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()
feature_names= dataset.columns[1:]
feature_important = pd.Series(data=model.feature_importances_, index = feature_names).sort_values(ascending=False).head(20)
plt.figure(2)
plt.barh(feature_important.index, feature_important)

plt.show()

LR= LinearRegression()
LR.fit(x,y)
lrimportance=LR.coef_  #w1-w4
LR.score(x_test,y_test)
plt.figure()
feature_names= dataset.columns[1:]
pyplot.bar([x for x in range(len(lrimportance))], lrimportance)
feature_important = pd.Series(data=LR.coef_, index = feature_names).sort_values(ascending=False).head(20)
plt.figure()
plt.barh(feature_important.index, feature_important)
plt.show()
# RFC = RandomForestClassifier()
# # fit the model
# RFC.fit(x, y.astype('string'))
# # get importance
# rfcimportance = RFC.feature_importances_
# # summarize feature importance
# for i,v in enumerate(rfcimportance):
#     print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# plt.figure(3)
# pyplot.bar([x for x in range(len(importance))], importance)
# pyplot.show()
# rfcfeature = pd.Series(data=RFC.feature_importances_, index = feature_names).sort_values(ascending=False).head(20)
# plt.figure(4)
# plt.barh(feature_important.index, feature_important)
#lr_svr=svm.SVR(kernel='linear')
#lr_svr.fit(x,y)

#lrfi=lr_svr.coef_.transpose()

#lr_svr.score(x_test,y_test)

