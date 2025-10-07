
"""
@File ：plot_test.py
@usage: --

@Author ：Colin
@Date ：2025/10/3 8:56
"""

import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt



X, y = load_iris(return_X_y= True)

fig = px.scatter_3d(y=X[:50, 0], color=X[:50, 1], z=X[:50, 2], x=X[:50,3], color_continuous_scale='Viridis')
fig.show()

print(X.shape)
