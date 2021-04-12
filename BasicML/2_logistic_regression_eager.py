# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# xy = pd.read_csv('data-03-diabetes.csv')
# x_train = xy.iloc[:, 0:-1]
# y_train = xy.iloc[:, [-1]]
# print(x_train.shape, y_train.shape)
# print(xy.head())

xy =np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]
print(x_train.shape, y_train.shape)
print(xy)

# Tensorflow Eager
# 위 Data를 기준으로 가설의 검증을 통해 Logistic Classification 모델을 만들도록 할 것.
# Tensorflow data API를 통해 학습시킬 값들을 담는다(Batch Size는 한번에 학습시킬 Size로 정함)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

# W와 b는 학습을 통해 생성되는 모델에 쓰이는 Weight와 Bias(초기값을 0이나 랜덤값으로 가능)
W = tf.Variable(tf.random.normal((8,1)), name='weight')
b = tf.Variable(tf.random.normal((1,)), name='bias')

def logistic_regression(features):
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) +b))
    return hypothesis

# Cost 함수 정의
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features))
                           + (1-labels) * tf.math.log(1-hypothesis))
    return cost
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 추론한 값은 0.5를 기준으로 작으면 0, 크면 1을 반환
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

# GradientTape를 통해 경사값을 계산합니다.
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features), features, labels)
    return tape.gradient(loss_value, [W,b])

# 학습실행
EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in iter(dataset):
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}"
                  .format(step, loss_fn(logistic_regression(features), features, labels)))

