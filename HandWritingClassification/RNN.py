# Created by [Yuexiong Ding] at 2018/6/6
# RNN 手写数字识别
#
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# 模型存储路径
ckpt_dir = './Ckpt_Dir/RNN'

# 超参设置
# 学习率
lr = 0.001
# 迭代次数
iteration = 1000
# 每轮训练数据的 batch_size = 128
batch_size = 128
# 每训练20次打印一次训练结果
display_step = 20

# 神经网络参数设置
# 输入层
n_inputs = 28
# 步数
n_steps = 28
# 隐层神经元个数
n_hidden_units = 128
# 输出类别个数
n_claeese = 10

# 输入数据集权重设置
# 输入(图片为28*28)
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_claeese])
# 权重
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_claeese]))
}
# 偏置值
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_claeese]))
}

# 计数器，当前训练轮数
global_step = tf.Variable(0, name='global_step', trainable=False)


# 定义 RNN 模型
def RNN(X, w, b):
    # (128, 28, 28) -> (128 * 28, 28)
    X = tf.reshape(X, [-1, n_inputs])

    # 进入隐层(128 * 28, 128)
    X_in = tf.matmul(X, w['in']) + b['in']
    # (128 * 28, 128) -> (128, 28, 128)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # LSTM 循环单元
    lstm_cell = rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32, time_major=False)

    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    return results


# 预测输出
pred = RNN(x, weights, biases)
# 代价
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# Adam 优化器
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 创建存储模型路径
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver = tf.train.Saver()

with tf.Session() as sess:
    # 读取数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 初始化变量
    tf.global_variables_initializer().run()

    # 获取模型检查点
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 当前训练步数
    start_step = global_step.eval()
    print('从第 %d 步开始训练...' % start_step)

    for i in range(start_step, iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
        if i % display_step == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            print('第 %d 次训练后准确率为 %f ' % (i, train_accuracy))

        # 更改计数器
        global_step.assign(i).eval()
        # 存储模型
        saver.save(sess, ckpt_dir + '/model.ckpt', global_step=global_step)

    test_xs = mnist.test.images
    test_ys = mnist.test.labels
    test_xs = test_xs.reshape([-1, n_steps, n_inputs])
    test_accuracy = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys})
    print('********************************************')
    print('模型最终准确率为 %f ' % test_accuracy)
