import tensorflow as tf
import numpy as np

X = np.array([1,2,3])
Y = np.array([1,2,3])


# Cost function in pure Python
# def cost_func_python(W,X,Y):
#     c = 0
#     for i in range(len(X)):
#         c += (W * X[i] - Y[i]) ** 2
#     return c/len(X)
#
# for feed_W in np.linspace(-3, 5,num=15):
#     curr_cost = cost_func_python(feed_W, X,Y)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
#
# print()
# -----------------------------------------------------------------------
# Cost function in TensorFlow
# def cost_func_tensor(W,X,Y):
#     hypothesis = X*W
#     return tf.reduce_mean(tf.square(hypothesis - Y))
# W_values = np.linspace(-3,5,num=15)
# cost_values = []
#
# for feed_W in W_values:
#     curr_cost = cost_func_tensor(feed_W, X,Y)
#     cost_values.append(curr_cost)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
#
# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (8,6)
# plt.plot(W_values, cost_values, "b")
# plt.ylabel('Cost(W')
# plt.xlabel('W')
# plt.show()
# ------------------------------------------------------------

# 현재 데이터 X와 Y에 대해 W가 1일 때 Cost가 가장 작다
# Cost가 최소가 되는 W를 어떻게 찾을 수 있는가
# => Gradient Descent

# Gradient descent 구현
# tf.random.set_seed(0) # for reproducibility
# x_data = [1., 2., 3., 4.]
# y_data = [1., 2., 3., 4.]

def operation(W):
    X = np.array([1, 2, 3])
    Y = np.array([1, 2, 3])
    for step in range(300):
        hypothesis = W * X
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        alpha = 0.01
        gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
        descent = W - tf.multiply(alpha, gradient)
        W.assign(descent)

        if step % 10 == 0:
            print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))

W = tf.Variable(tf.random.normal((1,), -100., 100.))
operation(W)

print(5.0 * W)
print(2.5 * W)

W = tf.Variable([5.0])
operation(W)