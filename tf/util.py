# coding=utf-8
import numpy as np

def gradient_test():
    print("一元梯度: 一阶导数")
    x = tf.constant(value=5.0)
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        tape.watch(x)
        y1 = 2*x
        y2 = x*x + 2
        y3 = x*x + 2 * x
    dy1_dx = tape.gradient(target=y1, sources=x)
    dy2_dx = tape.gradient(target=y2, sources=x)
    dy3_dx = tape.gradient(target=y3, sources=x)
    print("dy1_dx:", dy1_dx)
    print("dy2_dx:", dy2_dx)
    print("dy3_dx:", dy3_dx)

    print("二元梯度: 一阶导数")
    x = tf.constant(value=3.0)
    y = tf.constant(value=2.0)
    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        tape.watch([x, y])
        z1 = x*x*y + x*y
    dz1_dx = tape.gradient(target=z1, sources=x)
    dz1_dy = tape.gradient(target=z1, sources=y)
    dz1_d = tape.gradient(target=z1, sources=[x, y])
    print("dy1_dx:", dz1_dx)
    print("dy1_dx:", dz1_dy)
    print("dy1_d:", dz1_d)
    print("type dy1_d:", type(dz1_d))

def softmax(v):
    '''
    :param v: array
    :return: array of softmax probality
    '''
    expv = np.exp(v)
    v_sum = np.sum(expv)
    print(type(v), v)
    print(type(expv), expv)
    print(type(v_sum), v_sum)
    res = np.true_divide(expv, v_sum)
    print(type(res), res)
    return res


if __name__ == '__main__':
    #gradient_test()
    v = [9, 6, 3, 1]
    softmax(v)
