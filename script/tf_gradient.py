import tensorflow as tf
w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])
res = tf.matmul(w1, [[2],[1]])

#ys必须与xs有关，否则会报错
# grads = tf.gradients(res,[w1,w2])
#TypeError: Fetch argument None has invalid type <class 'NoneType'>

# grads = tf.gradients(res,[w1])
# # Result [array([[2, 1]])]

res2a=tf.matmul(w1, [[2],[1]])+tf.matmul(w2, [[3],[5]])
res2b=tf.matmul(w1, [[2],[4]])+tf.matmul(w2, [[8],[6]])

# grads = tf.gradients([res2a,res2b],[w1,w2])
#result:[array([[4, 5]]), array([[11, 11]])]

grad_ys=[tf.Variable([[1]]),tf.Variable([[2]])]
grads = tf.gradients([res2a,res2b],[w1,w2],grad_ys=grad_ys)
# Result: [array([[6, 9]]), array([[19, 17]])]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    re = sess.run(grads)
    print(re)
