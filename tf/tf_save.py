# coding=utf-8


from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
print('tf version:', tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
sample_num = 1000
train_labels = np.eye(10)[np.array([train_labels[:sample_num]]).reshape(-1)]
test_labels = np.eye(10)[np.array([test_labels[:sample_num-10]]).reshape(-1)]
train_images = train_images[:sample_num].reshape(-1, 28*28) / 255.0
test_images = test_images[:sample_num-10].reshape(-1, 28*28) / 255.0
print(type(train_images), type(train_labels), train_images.shape, train_labels.shape)
print(type(test_images), type(test_labels), test_images.shape, test_labels.shape)
print(train_labels[:10])


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# 每个epoch结束时更新模型
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True, # 只保存模型权重
    verbose=1,
    period=2  # 每5个epochs后保存模型
)
model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

new_model = create_model()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

new_model.load_weights(checkpoint_path)
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

print(model.predict(test_images)[0])

model.save_weights('./checkpoints/my_checkpoint')
model.save('my_model.h5')
new_model1 = keras.models.load_model('my_model.h5')
new_model1.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("new Restored model, accuracy: {:5.2f}%".format(100*acc))


