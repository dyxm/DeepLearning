# Created by Yuexiong Ding on 2018/5/14.
# Softmax Regression 手写数字识别
# 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取 mnist 数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 训练输入及label
x = tf.placeholder(tf.float32, [None, 784])
y_label = tf.placeholder(tf.float32, [None, 10])

# 参数w，b
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 模型输出
y_out = tf.nn.softmax(tf.matmul(x, w) + b)
# 交叉熵损失
loss = -tf.reduce_mean(y_label * tf.log(tf.clip_by_value(y_out, 1e-10, 1.0)))


# 参数优化
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 正确预测的结果
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_label, 1))
# 正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 创建Session，开始训练
with tf.Session() as sess:
    # 初始化所有变量
    tf.global_variables_initializer().run()

    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_x, y_label: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels})
        print('第 %d 轮训练后准确率为 %f' % (i, acc))






