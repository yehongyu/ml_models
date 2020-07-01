import tensorflow as tf

x_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9, 1]
y_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
x_2 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
y_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

x_placeholder = tf.placeholder(tf.float64, [10])
y_placeholder = tf.placeholder(tf.bool, [10])
auc = tf.metrics.auc(labels=y_placeholder, predictions=x_placeholder)
initializer = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    #sess.run(initializer)
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(2):
        auc_value, update_op = sess.run(auc, feed_dict={x_placeholder: x_1, y_placeholder: y_1})
        print('auc_1: ' + str(auc_value) + ", update_op: " + str(update_op))
        auc_value, update_op = sess.run(auc, feed_dict={x_placeholder: x_2, y_placeholder: y_2})
        print('auc_2: ' + str(auc_value) + ", update_op: " + str(update_op))

