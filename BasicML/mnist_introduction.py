# -*- coding : utf-8 -*-
# MNIST(Modified National Institute of Standards and Technology database
# 0~9까지의 손으로 쓴 숫자들로 이루어진 대형 데이터베이스
import numpy as np
import tensorflow as tf

# keras를 활용한 MNIST를 분류 모델 생성
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Tensorflow Keras
# 위 Data를 기준으로 분류 모델을 만듭니다.
# Tensorflow keras API를 통해 모델에 대한 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Adam Optimizer과 Cross Entropy Loss 선언
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5 Epoch로 학습할 Data로 학습 수행
model.fit(x_train, y_train, epochs=5)

print()

# 모델을 평가할 Test 데이터에 대한 정의
model.evaluate(x_test, y_test)
