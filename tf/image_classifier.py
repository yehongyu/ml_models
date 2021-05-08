#coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# Load mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(type(x_train), x_train.shape)
print(type(y_train), y_train.shape)
print(type(x_test), x_test.shape)
print(type(y_test), y_test.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

# Build nn network, (tf graph)
# Sequential: 线性堆叠layers的模型
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

# Set optimizer and los
# s
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
# param x: numpy array
# param y: numpy array
model.fit(x_train, y_train, batch_size=64, epochs=5)

# Evaluate
model.evaluate(x_test, y_test, verbose=2)



