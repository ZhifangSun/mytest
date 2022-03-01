from sklearn.datasets import make_blobs  # 导入数据集生成器
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN分类器
import matplotlib.pyplot as plt  # 导入画图工具
from sklearn.model_selection import train_test_split  # 导入数据集拆分工具
import numpy as np
data = make_blobs(n_samples=500, centers=2, n_features=2)# 生成样本数为500，有两个属性，五个分类的数据集

X, y = data# X保存数据属性，保存数据的分类
clf = KNeighborsClassifier()# 创建一个分类器
clf.fit(X, y) # 训练数据

# 以下代码用于画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z)

plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.spring, c=y, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
print(clf.score(X, y))
