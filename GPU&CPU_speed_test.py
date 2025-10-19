
"""
@File ：GPU&CPU_speed_test.py
@usage: --used to test the speed difference between GPU and CPU training

@Author ：Colin
@Date ：2025/10/3 8:56
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# 使用较大的网络规模
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(1000,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 生成随机数据
import numpy as np
x_train = np.random.rand(5000, 1000).astype('float32')
y_train = np.random.rand(5000, 1).astype('float32')

# 定义一个计时函数
def benchmark(device_name):
    with tf.device(device_name):
        start = time.time()
        model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=0)
        end = time.time()
    print(f"{device_name}: {end - start:.2f}s")

# 对比CPU与GPU训练时间
benchmark('/CPU:0')
benchmark('/GPU:0')

