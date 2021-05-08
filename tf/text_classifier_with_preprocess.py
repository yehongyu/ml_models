# coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
imdb = keras.datasets.imdb

print(tf.__version__)

vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
print(train_data.shape, test_data.shape)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(len(train_data[0]), len(train_data[1]))

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
print('word-index:', type(word_index), len(word_index))
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
print('word-index:', type(word_index), len(word_index))

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

# Build model struction
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16)) # 有多少个单词，每个单词用多少维
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

eval_size = 10000
x_val = train_data[:eval_size]
partial_x_train = train_data[eval_size:]

y_val = train_labels[:eval_size]
partial_y_train = train_labels[eval_size:]

history = model.fit(partial_x_train, partial_y_train, epochs=40,
                    batch_size=512,
                    validattion_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

