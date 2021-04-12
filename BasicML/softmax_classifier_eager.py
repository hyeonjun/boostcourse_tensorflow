# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)
# print(x_data, y_data)

# dataset을 선언합니다.
# dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
# dataset = dataset.repeat().batch(2)
nb_classes = 3 # class의 개수
# print(x_data.shape) # (8,4)
# print(y_data.shape) # (8,3)

# Weight and bias setting
W = tf.Variable(tf.random.normal((4,nb_classes)), name='weight')
b = tf.Variable(tf.random.normal((nb_classes,)), name='bias')
variables = [W,b]
# print(W,b)

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W) +b)
# print(hypothesis(x_data))

# Softmax onehot test
sample_db = [[8,2,1,4]]
sample_db = np.asarray(sample_db, dtype=np.float32)
# print(hypothesis(sample_db))

def cost_fn(X, Y):
    logits = hypothesis(X)
    cost = -tf.reduce_mean(Y * tf.math.log(logits), axis=1)
    cost_mean = tf.reduce_mean(cost)
    return cost_mean
# print(cost_fn(x_data, y_data))

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    y = x*x # x^2
dy_dx = g.gradient(y,x)
# print(dy_dx) # tf.Tensor(6.0, shape=(), dtype=float32)

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X,Y)
        grads = tape.gradient(loss, variables)
        return grads
# print(grad_fn(x_data, y_data))

def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads,variables))
        if (i==0) | ((i+1)%verbose==0):
            print('Loss at epoch %d: %f' %(i+1, cost_fn(X,Y).numpy()))
# fit(x_data, y_data)

# ---------------------------------------------------------------------

# Prediction Check
sample_data = [[2,1,3,2]] # answer_label [[0,0,1]]
sample_data = np.asarray(sample_data, dtype=np.float32)

a = hypothesis(sample_data)
# print(a)
# print(tf.argmax(a,1))

b = hypothesis(x_data)
# print(b)
# print(tf.argmax(b,1))
# print(tf.argmax(y_data,1))


# Convert as Class
class softmax_classifier(tf.keras.Model):
    def __init__(self, nb_classes):
        super(softmax_classifier, self).__init__()
        self.W = tf.Variable(tf.random.normal((4,nb_classes)), name='weight')
        self.b = tf.Variable(tf.random.normal((nb_classes,)), name='bias')

    def softmax_regression(self, X):
        return tf.nn.softmax(tf.matmul(X, self.W) + self.b)

    def cost_fn(self, X, Y):
        logits = self.softmax_regression(X)
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(logits), axis=1))
        return cost

    def grad_fn(self, X, Y):
        with tf.GradientTape() as tape:
            cost = self.cost_fn(x_data, y_data)
            grads = tape.gradient(cost, self.variables)
            return grads

    def fit(self, X,Y, epochs=2000, verbose=1000):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        for i in range(epochs):
            grads = self.grad_fn(X, Y)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i==0) | ((i+1)%verbose == 0) :
                print('Loss at epoch %d: %f' %(i+1, self.cost_fn(X,Y).numpy()))
# Loss at epoch 1: 2.352360
# Loss at epoch 1000: 0.227037
# Loss at epoch 2000: 0.150046

# Loss at epoch 1: 1.445905
# Loss at epoch 1000: 0.225392
# Loss at epoch 2000: 0.150181
model = softmax_classifier(nb_classes)
model.fit(x_data, y_data)






