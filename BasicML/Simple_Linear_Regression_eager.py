# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
x_data = np.arange(1,6,1)
y_data = np.arange(1,6,1)
print(x_data, y_data)

import matplotlib.pyplot as plt

v = np.arange(1., 5., 1.)
print(v)
print(tf.reduce_mean(v))
print(tf.square(3))

W = tf.Variable(2.0)
b = tf.Variable(0.5)
print(W.numpy(), b.numpy())
hypothesis = W * x_data + b
print(hypothesis.numpy())

# plt.plot(x_data, hypothesis.numpy(), 'r-')
# plt.plot(x_data, y_data, 'o')
# plt.ylim(0,10) # y축 범위
# plt.show()



# Cost
# reduce_mean : 평균, square : 제곱
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
with tf.GradientTape() as tape:
    hypothesis = W * x_data + b
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

W_grad, b_grad = tape.gradient(cost, [W, b])
print(W_grad.numpy(), b_grad.numpy())
# 25.0 7.0

learning_rate = 0.01
W.assign_sub(learning_rate * W_grad)
b.assign_sub(learning_rate * b_grad)
print(W.numpy(), b.numpy())
# 1.75 0.43

# plt.plot(x_data, hypothesis.numpy(), 'r-')
# plt.plot(x_data, y_data, 'o')
# plt.ylim(0,8)
# plt.show()



# 여러 번 반복시키기
W = tf.Variable(0.1)
b = tf.Variable(0.5)
for i in range(1000):
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i%10==0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

print()

# predict
print(W * 5 + b)
print(W * 2.5 + b)

plt.plot(x_data, y_data, 'o')
plt.plot(x_data, hypothesis.numpy(), 'r-')
plt.ylim(0,10)
plt.show()