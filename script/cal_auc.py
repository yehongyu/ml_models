import tensorflow as tf

def cal_auc(prediction_list, label_list):
    prediction_tensor = tf.convert_to_tensor(prediction_list)
    label_tensor = tf.convert_to_tensor(label_list)
    auc_value, auc_op = tf.metrics.auc(label_tensor[:,1], prediction_tensor[:,1], num_thresholds=2000)
    initializer = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    res = None
    with tf.Session() as sess:
        sess.run(initializer)
        sess.run(auc_op)
        res = sess.run(auc_value)

        print(prediction_tensor)
        print(label_tensor)
        print("AUC:" + str(res))
    return res

def cal_auc_with_placeholder(x, y):
    x_placeholder = tf.placeholder(tf.float64, [10])
    y_placeholder = tf.placeholder(tf.bool, [10])
    auc_value, auc_op = tf.metrics.auc(labels=y_placeholder, predictions=x_placeholder)
    initializer = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(initializer)
        sess.run(auc_op, feed_dict={x_placeholder: x, y_placeholder: y})
        res = sess.run(auc_value, feed_dict={x_placeholder: x, y_placeholder: y})
        print("placeholder AUC:" + str(res))
    return res

'''
x_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9, 1]
y_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
cal_auc(x_1, y_1)
cal_auc_with_placeholder(x_1, y_1)
'''

x_2 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
y_2 = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
#cal_auc(x_2, y_2)
cal_auc_with_placeholder(x_2, y_2)

x_3 = [[0, 1], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
y_3 = [[0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
cal_auc(x_3, y_3)
