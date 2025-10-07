"""
@File ：tests222.py
@usage: --

@Author ：Colin
@Date ：2025/10/6 1:04
"""
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import plotly.express as px

X, y = load_iris(return_X_y= True)
fig = px.parallel_coordinates(X,
                              color_continuous_scale=px.colors.qualitative, # 可以选择不同的颜色方案
                              title="四维数据平行坐标图 (150 行)",
                              labels={"Feature_A": "特性 A", "Feature_B": "特性 B",
                                      "Feature_C": "特性 C", "Feature_D": "特性 D"},
                              # color="Feature_A" # 也可以用某一列的值来着色，例如，根据Feature_A的值来着色
                             )

fig.show()
