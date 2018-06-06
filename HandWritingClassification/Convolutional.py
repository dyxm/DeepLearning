# Created by Yuexiong Ding on 2018/5/14.
# 两层卷积神经网络 手写数字识别
#
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def init_variable(shape, is_filter=True):
    """初始化参数"""
    if is_filter:
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    else:
        return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, w):
    """卷积操作"""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """最大池化"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def act_function(x, act_type='relu'):
    """激活函数"""
    if act_type == 'relu':
        return tf.nn.relu(x)
    elif act_type == 'tanh':
        return tf.nn.tanh(x)
    elif act_type == 'sigmoid':
        return tf.nn.sigmoid(x)


def variable_summaries(var):
    """对一个张量添加多个摘要描述"""
    with tf.name_scope('summaries'):
        # 均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # 标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():
    # 模型存储地址
    ckpt_dir = "./Ckpt_Dir/SoftmaxRegression"

    # 训练日志存储地址
    log_dir = "./Mnist_Logs/SoftmaxRegression"

    # 计数器变量，当前第几步
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # 读取数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 训练样本及label
    x = tf.placeholder(tf.float32, [None, 784])
    y_label = tf.placeholder(tf.float32, [None, 10])

    # dropout率
    keep_prob = tf.placeholder(tf.float32)

    # 将图片还原为 28*28 矩阵
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积层
    w_conv1 = init_variable([5, 5, 1, 32], is_filter=True)
    b_conv1 = init_variable([32], is_filter=False)
    # 激活
    h_conv1 = act_function(conv2d(x_image, w_conv1) + b_conv1)
    # 池化
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积层
    w_conv2 = init_variable([5, 5, 32, 64])
    b_conv2 = init_variable([64], is_filter=False)
    h_conv2 = act_function(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 第一层全连接
    w_fc1 = init_variable([7 * 7 * 64, 1024])
    b_fc1 = init_variable([1024], is_filter=False)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = act_function(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    # 随机参数失活
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 第二层全连接层 即输出层
    w_fc2 = init_variable([1024, 10])
    b_fc2 = init_variable([10], is_filter=False)
    y_logit = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    # 交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_logit))
    # 可视化
    tf.summary.scalar('cross_entropy', cross_entropy)

    # 参数优化
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_logit, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #
    tf.summary.scalar('accuracy', accuracy)

    # 创建存储模型路径
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    saver = tf.train.Saver()



    with tf.Session() as sess:
        # 训练次数
        iteration = 2000

        # summaries合并
        merged_summary_op = tf.summary.merge_all()
        # 写到指定的磁盘路径中
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

        # 初始化变量
        tf.global_variables_initializer().run()

        # 获取模型检查点
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 模型当前训练步数
        global_start_step = global_step.eval()
        print('从第 %d 步开始训练...' % global_start_step)

        for i in range(global_start_step, iteration):
            batch_x, batch_y = mnist.train.next_batch(50)
            summary, _ = sess.run([merged_summary_op, train_step],
                                  feed_dict={x: batch_x, y_label: batch_y, keep_prob: 0.5})
            train_writer.add_summary(summary, i)

            # 每训练10次，打印模型准确率
            if i % 10 == 0:
                train_accuracy, summary = sess.run([accuracy, merged_summary_op],
                                                   feed_dict={x: batch_x, y_label: batch_y, keep_prob: 1.0})
                print('第 %d 次训练，模型准确率为 %f' % (i, train_accuracy))

            # 更新计数器
            global_step.assign(i).eval()
            # 存储模型
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0})
        print('********************************************')
        print('模型在测试集上的准确率为 %f' % test_accuracy)

        train_writer.close()


if __name__ == '__main__':
    main()
