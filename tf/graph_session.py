#coding=utf-8
import tensorflow as tf

def process_tf1():
    g = tf.Graph() # 初始化计算图
    with g.as_default():
        a = tf.constant([[10, 10], [11, 1]])
        x = tf.constant([[1, 0], [0, 1]])
        b = tf.Variable(0)
        y = tf.matmul(a, x) + b
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=g) as sess:
        #print(init_op)
        sess.run(init_op)
        print(sess.run(y))

def process_tf2():
    print('eager mode:', tf.executing_eagerly())
    a = tf.constant([[10, 10], [11, 1]])
    x = tf.constant([[1, 0], [0, 1]])
    b = tf.Variable(0)
    y = tf.matmul(a, x) + b
    print(y.numpy())

#process_tf1()

tf.enable_eager_execution()
process_tf2()
