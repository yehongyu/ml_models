#coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print('tf version:', tf.__version__)

# Load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print('train shape:', type(train_images), train_images.shape, train_labels.shape)
print('train shape:', type(test_images), test_images.shape, test_labels.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data normalization, 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0
# conv2D
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Build network
model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(28, 28)), #28*28=784，没有参数需要学习
    keras.layers.Conv2D(32, 2, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'), #input=784, output=128
    keras.layers.Dense(10, activation='softmax') #input=128, output=10
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train
model.fit(train_images, train_labels, batch_size=64, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# predict
predictions = model.predict(test_images)
print(np.argmax(predictions[0]), test_labels[0], predictions[0])

