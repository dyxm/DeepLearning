# Created by [Yuexiong Ding] at 2018/6/7
# 自编码器实现手写字体识别
#
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 模型存储路径
ckpt_dir = './Ckpt_Dir/AutoEncoder'

# 超参设置
# 学习率
lr = 0.01
# 训练次数
iteration = 10000
# 训练批次的大小
batch_size = 256
# 每训练几次打印一次损失结果
display_cost_step = 1
# 每训练几次打印自编码后的图片
display_img_step = 9999
# 最终用于测试模型的图片数量
example_num = 10

# 网络参数设置
# 输入层特征值个数
n_input = 784
# 第一隐层的神经元个数
n_hidden_1 = 256
# 第二隐层神经元个数
n_hidden_2 = 128

# 输入设置
X = tf.placeholder(tf.float32, [None, n_input])

# 训练步数计数器
global_step = tf.Variable(0, name='global_step', trainable=False)

# 权重设置
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}


def encoder(x):
    """压缩函数"""
    # 一层
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1'])
    # 第二层
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['encoder_h2']) + biases['encoder_b2'])
    return layer_2


def decoder(x):
    """解压函数"""
    # 一层
    layer_1 = tf.sigmoid(tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1'])
    # 二层
    layer_2 = tf.sigmoid(tf.matmul(layer_1, weights['decoder_h2']) + biases['decoder_b2'])
    return layer_2


# 构建模型
# 压缩
encoder_op = encoder(X)
# 解压(输出预测值)
decoder_op = decoder(encoder_op)

# 代价
cost = tf.reduce_mean(tf.pow(X - decoder_op, 2))
# 优化
train_op = tf.train.RMSPropOptimizer(lr).minimize(cost)

# 新建模型存储路
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver()


# 开启session
with tf.Session() as sess:
    # 读取数据集
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # 初始化变量
    tf.global_variables_initializer().run()

    # 加载已有模型
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 当前训练步数
    start_step = global_step.eval()
    print('从第 %d 步开始训练...' % start_step)

    # 训练
    for i in range(start_step, iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c = sess.run([train_op, cost], feed_dict={X: batch_xs})

        # 打印损失
        if i % display_cost_step == 0:
            print('第 %d 次训练后损失为 %f ' % (i, c))

        # 更改计数器
        global_step.assign(i).eval()
        # 保存模型
        saver.save(sess, ckpt_dir + '/model.ckpt', global_step=global_step)

        # 测试
        if i % display_img_step == 0:
            # 样例图
            examples_xs = mnist.test.images[: example_num]
            encode_decode = sess.run(decoder_op, feed_dict={X: examples_xs})

            # 显示图片
            f, a = plt.subplots(2, example_num)
            for j in range(example_num):
                a[0][j].imshow(np.reshape(examples_xs[j], (28, 28)))
                a[1][j].imshow(np.reshape(encode_decode[j], (28, 28)))
            f.show()
            plt.draw()
            plt.waitforbuttonpress()


