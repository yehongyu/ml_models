# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

model = Model()
print('init predict:', model(3.0).numpy())

epochs = 30
for epoch in range(epochs):
    train(model, inputs, outputs, learning_rate=0.1)
    current_loss = loss(model(inputs), outputs)
    print('%s:loss=%s, W=%s, b=%s' % (epoch, current_loss.numpy(), model.W.numpy(), model.b.numpy()))


