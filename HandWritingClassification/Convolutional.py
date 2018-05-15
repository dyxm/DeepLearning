# Created by Yuexiong Ding on 2018/5/14.
# 两层卷积神经网络 手写数字识别
# 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable

# 读取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 训练样本及label
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])

# 将图片还原为 28*28 矩阵
x_image = tf.reshape(x, [-1, 28, 28, 1])




