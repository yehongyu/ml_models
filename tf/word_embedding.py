# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

embedding_layer = tf.keras.layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1, 2, 3]))
print(result.numpy())

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
print(result.shape)

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
encoder.subwords[:20]

padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)

train_batch, train_labels = next(iter(train_batches))
print(train_batch.numpy())
print('vocab size:', encoder.vocab_size)

embedding_dim=16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    train_batches, epochs=10,
    validation_data=test_batches, validation_steps=20
)

print('model layers:', len(model.layers))
e = model.layers[0]
print('e trainable var:', e.trainable_variables)
print('e var:', e.variables)
weights = e.get_weights()[0]
print('e weights shape:', weights.shape)

print('dense var:', (model.layers[3]).variables)
print('dense trainable var:', (model.layers[3]).trainable_variables)
