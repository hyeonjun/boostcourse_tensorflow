# -*- coding : utf-8 -*-
# linear regresstion에 Normalization, Learing Decay, L2_loss를 적용
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 정규화를 위한 함수 (최대 최소값이 1과 0이 되도록 Scaling 한다)
def normalization(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator/denominator

# Data
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# x_train = xy[:,0:-1]
# y_train = xy[:,[-1]]
# plt.plot(x_train, 'ro')
# plt.plot(y_train)
# plt.show()
# ---------------------------------------------------------------

xy = normalization(xy)
x_train = xy[:,0:-1]
y_train = xy[:,[-1]]
# plt.plot(x_train, 'ro')
# plt.plot(y_train)
# plt.show()
# ---------------------------------------------------------------

# 위 Data를 기준으로 Linear Regression 모델을 만들도록 하겠습니다.
# Tensorflow data API를 통해 학습시킬 값들을 담는다 (Batch Size는 한번에 학습시킬 Size)
# X(features), Y(labels)는 실제 학습에 쓰일 Data (연산을 위해 Type을 맞춰준다)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

# W와 b는 학습을 통해 생성되는 모델에 쓰이느 Weight와 Bias (초기값을 Variable : 0이나 Random값으로 가능
# tf.random_normal/tf.zeros)
W = tf.Variable(tf.random.normal((4, 1)), dtype=tf.float32)
b = tf.Variable(tf.random.normal((1,)), dtype=tf.float32)

# Linear Regression의 Hypothesis를 정의한다(y=Wx+b)
def linearReg_fn(features):
    hypothesis = tf.matmul(features, W) + b
    return hypothesis


# L2 loss를 적용할 함수를 정의합니다.
# Weight 수가 많아지면 수만큼 더한다. tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
def l2_loss(loass, beta=0.01):
    W_reg = tf.nn.l2_loss(W) # output = sum(t**2)/2
    loss = tf.reduce_mean(loss + W_reg * beta)
    return loss

# 가설을 검증할 Cost 함수를 정의합니다.(Mean Square Error)
def loss_fn(hypothesis, features, labels, flag=False):
    cost = tf.reduce_mean(tf.square(hypothesis - labels))
    if(flag):
        cost = l2_loss(cost)
    return cost


# Learning Rate값을 조정하기 위한 Learning Decay 설정
# 5개 파라미터 설정
# - starter_learning_rate : 최초 학습 시 사용될 learning rate (0.1로 설정하여 0.96씩 감소하는지 확인)
# - global_step : 현재 학습 횟수
# - 1000 : 곱할 횟수 정의(1000번마다 적용)
# - 0.96 : 기존 learning에 곱할 값
# - 적용유무 devayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
# - decayed_learning_rate = learning_rate * decay ^ (global_step / decay_steps)
is_decay = True
starter_learning_rate = 0.1
if(is_decay):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=starter_learning_rate,
        decay_steps=50,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
else:
    optimizer - tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)

def grad(hypothesis, features, labels, l2_flag):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(linearReg_fn(features), features, labels, l2_flag)
    return tape.gradient(loss_value, [W,b]), loss_value


# TensorFlow를 통해 학습을 진행합니다.
EPOCHS = 10001
for step in range(EPOCHS):
    for features, labels in dataset:
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        grads, loss_value = grad(linearReg_fn(features), features, labels, False)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
    if step % 10 == 0:
        print("Iter: {}, Loss: {:.4f}".format(step, loss_value))
