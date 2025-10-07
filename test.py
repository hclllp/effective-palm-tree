"""
@File ：test.py
@usage: --

@Author ：Colin
@Date ：2025/10/6 1:02
"""

from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt  # 导入matplotlib


iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)  # 生成决策树图像
plt.show()  # 显示图像
