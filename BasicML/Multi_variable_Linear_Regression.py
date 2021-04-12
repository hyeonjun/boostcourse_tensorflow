# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# x1_data = [1,0,3,0,5]
# x2_data = [0,2,0,4,0]
# y_data = [1,2,3,4,5]
#
# W1 = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))
# W2 = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))
# b = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))
#
# learning_rate = tf.Variable(0.001)
#
# for i in range(1000+1):
#     with tf.GradientTape() as tape:
#         hypothesis = W1 * x1_data + W2 * x2_data + b
#         cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#     W1_grad, W2_grad, b_grad = tape.gradient(cost, [W1, W2, b])
#     W1.assign_sub(learning_rate * W1_grad)
#     W2.assign_sub(learning_rate * W2_grad)
#     b.assign_sub(learning_rate * b_grad)
#
#     if i % 50 == 0:
#         print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
#             i, cost.numpy(), W1.numpy()[0], W2.numpy()[0], b.numpy()[0]
#         ))


# --------------------------------------------------------------------------
# Simple Example (2 variables with Matrix)
# x_data = [
#     [1., 0., 3., 0., 5.],
#     [0., 2., 0., 4., 0.]
# ]
# y_data  = [1, 2, 3, 4, 5]
#
# W = tf.Variable(tf.random.uniform((1,2), -1.0, 1.0))
# # print(W)
# # <tf.Variable 'Variable:0' shape=(1, 2) dtype=float32,
# # numpy=array([[-0.46509457, -0.686527  ]], dtype=float32)>
# b = tf.Variable(tf.random.uniform((1,), -1.0, 1.0))
# learning_rate = tf.Variable(0.001)
# # print(learning_rate)
# # <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.001>

# for i in range(1000+1):
#     with tf.GradientTape() as tape:
#         hypothesis = tf.matmul(W, x_data) + b # (1,2) * (2,5) = (1,5)
#         cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#
#         W_grad, b_grad = tape.gradient(cost, [W,b])
#         W.assign_sub(learning_rate * W_grad)
#         b.assign_sub(learning_rate * b_grad)
#
#     if i % 50 == 0:
#         print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
#             i, cost.numpy(), W.numpy()[0][0], W.numpy()[0][1], b.numpy()[0]))
# --------------------------------------------------------------------------

# Hypothesis without b
# 위 코드에서 bias(b)를 행렬에 추가
# x_data = [
#     [1.,1.,1.,1.,1.], # bias(b)
#     [1.,0.,3.,0.,5.],
#     [0.,2.,0.,4.,0.]
# ]
# y_data = [1,2,3,4,5]
#
# W = tf.Variable(tf.random.uniform((1,3), -1.0, 1.0)) # [1,3]으로 변경하고, b 삭제
# learning_rate = 0.001
# optimizer = tf.keras.optimizers.SGD(learning_rate)
#
# for i in range(1000+1):
#     with tf.GradientTape() as tape:
#         hypothesis = tf.matmul(W, x_data) # b가 없음
#         cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#
#     grads = tape.gradient(cost, [W])
#     optimizer.apply_gradients(grads_and_vars=zip(grads, [W]))
#     if i % 50 == 0:
#         print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.4f}".format(
#             i, cost.numpy(), W.numpy()[0][0], W.numpy()[0][1], W.numpy()[0][2]))
# --------------------------------------------------------------------------


# Custom Gradient
# tf.train.GradientDescentOptimizer() : optimizer
# optimizer.apply_gradients() : update
# Multi-variable linear regression (1)
# X = tf.constant([[1., 2.],
#                  [3., 4.],])
# # print(X)
# # tf.Tensor(
# # [[1. 2.]
# #  [3. 4.]], shape=(2, 2), dtype=float32)
# y = tf.constant([[1.5], [3.5]])
# # print(y)
# # tf.Tensor(
# # [[1.5]
# #  [3.5]], shape=(2, 1), dtype=float32)
# W = tf.Variable(tf.random.normal((2,1)))
# b = tf.Variable(tf.random.normal((1,)))
#
# # Create an optimizer
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#
# n_epoch = 1000+1
# print("epoch | cost")
# for i in range(n_epoch):
#     # use tf.GradientTape() to record the gradient of the cost function
#     with tf.GradientTape() as tape:
#         y_pred = tf.matmul(X,W) + b
#         cost = tf.reduce_mean(tf.square(y_pred - y))
#
#     # calculates the gradients of the loss
#     grads = tape.gradient(cost, [W,b])
#
#     # updates parameters (W and b)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
#     if i % 50 == 0:
#         print("{:5} | {:10.6f}".format(i, cost.numpy()))
# --------------------------------------------------------------------------


# Predicting exam score
# regression using three inputs (x1, x2, x3)
# x1 = [73.,93.,89.,96.,73.]
# x2 = [80.,88.,91.,98.,66.]
# x3 = [75.,93.,90.,100.,70.]
# Y = [152.,185.,180.,196.,142.]
#
# # weights
# w1 = tf.Variable(10.)
# w2 = tf.Variable(10.)
# w3 = tf.Variable(10.)
# b = tf.Variable(10.)
#
# learning_rate = 0.000001
#
# for i in range(1000+1):
#     with tf.GradientTape() as tape:
#         hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
#         cost = tf.reduce_mean(tf.square(hypothesis - Y))
#     w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1,w2,w3,b])
#
#     # update w1,w2,w3 and b
#     w1.assign_sub(learning_rate * w1_grad)
#     w2.assign_sub(learning_rate * w2_grad)
#     w3.assign_sub(learning_rate * w3_grad)
#     b.assign_sub(learning_rate * b_grad)
#
#     if i % 50 == 0:
#         print("{:5} | {:12.4f}".format(i, cost.numpy()))
# --------------------------------------------------------------------------


# Multi-variable linear regression(1)
# random 초기화 : tf.random.normal()

# # data and label
# x1 = [ 73.,  93.,  89.,  96.,  73.]
# x2 = [ 80.,  88.,  91.,  98.,  66.]
# x3 = [ 75.,  93.,  90., 100.,  70.]
# Y  = [152., 185., 180., 196., 142.]
#
# # random weights
# w1 = tf.Variable(tf.random.normal((1,)))
# w2 = tf.Variable(tf.random.normal((1,)))
# w3 = tf.Variable(tf.random.normal((1,)))
# b  = tf.Variable(tf.random.normal((1,)))
#
# learning_rate = 0.000001
#
# for i in range(1000+1):
#     with tf.GradientTape() as tape:
#         hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
#         cost = tf.reduce_mean(tf.square(hypothesis - Y))
#     w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1,w2,w3,b])
#
#     # update w1,w2,w3 and b
#     w1.assign_sub(learning_rate * w1_grad)
#     w2.assign_sub(learning_rate * w2_grad)
#     w3.assign_sub(learning_rate * w3_grad)
#     b.assign_sub(learning_rate * b_grad)
#
#     if i % 50 == 0:
#         print("{:5} | {:12.4f}".format(i, cost.numpy()))
# --------------------------------------------------------------------------


# Multi-varialbe linear regression (2)
# Matrix 사용

data = np.array([
    # X1,  X2,  X3,  Y
    [ 73., 80., 75., 152.],
    [ 93., 88., 93., 185.],
    [ 89., 91., 90., 180.],
    [ 96., 98., 100., 196.],
    [ 73., 66., 70., 142.]
], dtype=np.float32)

# slice data
X = data[:, :-1]
y = data[:, [-1]]
W = tf.Variable(tf.random.normal((3,1)))
b = tf.Variable(tf.random.normal((1,)))
learning_rate = 0.000001

# hypothesis, prediction function
def predict(X):
    return tf.matmul(X, W) + b
print("epoch | cost")
n_epochs = 2000
for i in range(n_epochs):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - y)))

    W_grad, b_grad = tape.gradient(cost, [W,b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))

print(W.numpy())
# [[0.42570975]
#  [1.1699803 ]
#  [0.41924384]]
print(b.numpy())
# [-0.05497888]
print(tf.matmul(X, W) + b)
# tf.Tensor(
# [[156.06354]
#  [181.48396]
#  [182.03336]
#  [197.39561]
#  [137.5876 ]], shape=(5, 1), dtype=float32)

# predict
print(y) # labels, 실제값
# [[152.]
#  [185.]
#  [180.]
#  [196.]
#  [142.]]
print(predict(X).numpy()) # prediction, 예측값
# [[149.52518]
#  [185.88026]
#  [179.89478]
#  [196.72794]
#  [142.65729]]

# 새로운 데이터에 대한 예측
print(predict([[89., 95., 92.],[84., 92., 85.]]).numpy())
# [[183.95416]
#  [174.14842]]








