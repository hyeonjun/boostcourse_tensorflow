# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Data
# x_data가 2차원 배열이기에 2차원 공간에 표현하여 x1과 x2를 기준으로
# y_data 0과 1로 구분하는 예제
# 붉은색과 푸른색으로 0과 1을 표시
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
# plt.scatter(x_data[0][0],x_data[0][1], c='red', marker='^')
# plt.scatter(x_data[3][0],x_data[3][1], c='red', marker='^')
# plt.scatter(x_data[1][0],x_data[1][1], c='blue', marker='^')
# plt.scatter(x_data[2][0],x_data[2][1], c='blue', marker='^')
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()


# Tensorflow Eaget
# 위 Data를 기준으로 XOR처리를 위한 모델을 만들도록 하겠습니다.
# Tensorflow data API를 통해 학습시킬 값들을 담는다(Batch Size는 한번에 학습시킬 Size로 정함)
# Preprocess function으로 features, labels는 실재 학습에 쓰일 Data 연산을 위해 Type를 맞춤
dataset = tf.data.Dataset.from_tensor_slices((x_data,y_data)).batch(len(x_data))

def preprocess_data(features, labels):
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    return features,labels

# 1) Logistic Regression으로 XOR 모델을 만듬
# W와 b는 학습을 통해 생성되는 모델에 쓰이는 Weight와 Bias(초기값으로 Variable 0이나 Random값으로 가능)
# 랜덤 -> tf.random_normal([2,1])
W = tf.Variable(tf.zeros((2,1)), name='weight')
b = tf.Variable(tf.zeros((1,)), name='bias')
# print("W={}, B={}".format(W.numpy(), b.numpy()))


# Sigmoid 함수를 가설로 선언
# Sigmoid는 아래 그래프와 같이 0과 1의 값만을 리턴. tf.sigmoid(tf.matmul(X,W)+b)
def logistic_regression(features):
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + b))
    return hypothesis

# Cost 함수
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features)) +
                           (1-labels) * tf.math.log(1-hypothesis))
    return cost

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


# 추론한 값은 0.5를 기준으로(Sigmoid 그래프 참조)로 0과 1의 값을 리턴합니다
# Sigmoid 함수를 통해 예측값이 0.5보다 크면 1을 반환, 보다 작으면 0으로 반환
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.float32))
    return accuracy

# GradientTape를 통해 경사값을 계산합니다.
def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape :
        loss_value = loss_fn(logistic_regression(features), features, labels)
    return tape.gradient(loss_value, [W,b])

# Tensorflow를 통한 실행을 위해 Session 선언
# Cost함수를 통해 학습시켜 모델 생성
EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in dataset:
        features, labels = preprocess_data(features, labels)
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(logistic_regression(features), features, labels)))

print("W = {}, B = {}".format(W.numpy(), b.numpy()))
x_data, y_data = preprocess_data(x_data, y_data)
test_acc = accuracy_fn(logistic_regression(x_data),y_data)
print("Testset Accuracy: {:.4f}".format(test_acc))