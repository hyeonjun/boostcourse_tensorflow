# -*- coding : utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 학습에 쓰이는 Data
# 50,000 movie reviews from the internet Movie Database(10000개의 빈도수가 높은 단어를
# 학습 시 Vector에 사용)
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print("Training entried : {}, labels : {}".format(len(train_data), len(train_labels)))
# print(train_data[0])


# IMDB Data를 Vector을 실제 값으로 변환하여 출력
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# print(decode_review(train_data[4]))
# print(train_labels[4])

# Tensorflow Keras
# 위 Data를 기준으로 분류 모델을 만들도록 하겠습니다
# 학습과 평가를 위해 동일길이인 256길이의 단어로 PAD 값을 주어 맞춰줌(뒤의 길이는 0값으로 맞춰줌)
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)
# print(len(train_data[0]), len(test_data[0]))
# print(train_data[0])

# Tensorflow Keras API를 통해 모델에 대한 정의
# 입력 Size와 학습시킬 Layer의 크기와 Activation Function 정의
# input shape is the vocabulary count used for the movie reviews(10,000 words)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# Adam Optimizer과 Cross Entropy Loss 선언
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델을 평가할 Test 데이터에 대한 정의(10000을 기준으로 학습과 평가 수행)
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

print()

result = model.evaluate(test_data, test_labels)
print(result)