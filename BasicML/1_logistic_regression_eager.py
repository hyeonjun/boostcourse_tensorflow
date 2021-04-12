# -*- coding:utf-8 -*-
# Logistic Classification은 True or False와 같은 Binary나 복수개의 다항 분류에 쓰입니다.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# x_data가 2차원 배열이기에 2차원 공간에 표현하여 x1과 x2를 기준으로 y_data 0과 1로 구분합니다.
# Logistic Classification 통해 보라색과 노란색 y_data(Label)을 구분
# Test 데이터는 붉은 색의 위치와 같이 추론 시 1의 값을 가지게 된다


x_train = [[1., 2.],
          [2., 3.],
          [3., 1.],
          [4., 3.],
          [5., 3.],
          [6., 2.]]
y_train = [[0.],
          [0.],
          [0.],
          [1.],
          [1.],
          [1.]]

x_test = [[5.,2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train] # 1, 2, 3, ..
x2 = [x[1] for x in x_train] # 2, 3, 1, ..

# colors = [int(y[0] % 3) for y in y_train]
# plt.scatter(x1, x2, c=colors, marker='^')
# plt.scatter(x_test[0][0],  x_test[0][1], c="red")
#
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

# Tensorflow Eager
# 위 data를 기준으로 가설의 검증을 통해 Logistic Classification 모델을 만듬
# tensorflow data API를 통해 학습시킬 값들을 담는다.(Batch Size는 한번에 학습시킬 Size로 정함)
# features, labels는 실재 학습에 쓰일 Data(연산을 위해 Type을 맞춰줘야함)
# [_, _] [_] 인 데이터가 6개 만들어질 것
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
# .repeat()

# W와 b는 학습을 통해 생성되는 모델에 쓰이는 Weight와 Bias(초기값을 Variable : 0이나
# Random값으로 가능(tf.random.normal([2,1]))
W = tf.Variable(tf.zeros([2,1]), name='weight') # 0으로 초기값 설정
b = tf.Variable(tf.zeros([1]), name='bias')


# Sigmoid 함수를 가설로 선언합니다.
# Sigmoid는 아래 그래프와 같이 0과 1의 값만을 리합니다.
# tf.sigmoid(tf.matmul(X,W)+b)
def logistic_regression(features):
#                           1 / (1+ e^(-x)) // x => tf.matmul(features, W) + b
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis

# 가설을 검증할 Cost함수를 정의합니다.
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features)) +
                           (1-labels) * tf.math.log(1-hypothesis))
    return cost

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 추론한 값은 0.5를 기준(Sigmoid 그래프 참조)로 0과 1의 값을 리턴
# sigmoid 함수를 통해 예측값이 0.5보다 크면 1을 반환, 0.5보다 작으면 0을 반환
# 가설을 통해 실제값과 비교한 정확도를 측정
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

# GradientTape를 통해 경사값을 계산합니다.
def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value=loss_fn(logistic_regression(features), features, labels)
    return tape.gradient(loss_value, [W,b])

# 학습을 실행합니다.
# 위의 Data를 Cost함수를 통해 학습시킨 후 모델을 생성합니다.
# 새로운 Data를 통한 검증 수행 [5,2]의 Data로 테스트 수행 (그래프상 1이 나와야 정상!)
n_epochs = 1000
for step in range(n_epochs +1):
    for features, labels, in iter(dataset):
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 100 == 0:
            print("iter: {}, Loss: {:.4f}"
                  .format(step,loss_fn(logistic_regression(features),
                                       features, labels)))
test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("Testset Accuracy : {:.4f}".format(test_acc))