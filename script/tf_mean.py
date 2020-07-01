import numpy as np
import tensorflow as tf

loss = [7.2610840e+01, 0.0000000e+00, 3.7714744e+07, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]
np_mean = np.mean(loss)
np_sum = np.sum(loss)
print("np_mean=", np_mean)
print("np_sum=", np_sum)

loss_tensor = tf.Variable(loss)

reduce_sum_loss_tensor = tf.reduce_sum(loss_tensor)
reduce_mean_loss_tensor = tf.reduce_mean(loss_tensor)

mean_loss_val_tensor, mean_loss_op_tensor = tf.compat.v1.metrics.mean(loss_tensor)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.reduce_mean(loss_tensor)))
    val, op = tf.compat.v1.metrics.mean(loss_tensor)
    print(sess.run(op))
    print('mean_loss_op:', sess.run(mean_loss_op_tensor))

    print('mean_loss_val:', sess.run(mean_loss_val_tensor))
    print('reduce_mean_loss:', sess.run(reduce_mean_loss_tensor))
    print('reduce_sum_loss:', sess.run(reduce_sum_loss_tensor))

    print('2 mean_loss_op:', sess.run(mean_loss_op_tensor))
    print('2 mean_loss_val:', sess.run(mean_loss_val_tensor))
    print('2 reduce_mean_loss:', sess.run(reduce_mean_loss_tensor))
    print('2 reduce_sum_loss:', sess.run(reduce_sum_loss_tensor))
