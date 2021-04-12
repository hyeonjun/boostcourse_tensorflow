# -*- coding : utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnis = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnis.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Fashion MNIST Data 확인 - 4번째 배열 드레스
# plt.figure()
# plt.imshow(train_images[3])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# Tensorflow Keras
# 위 Data를 기준으로 분류 모델을 만듬
# 0~1 사이의 값으로 정규화 및 Data 출력
train_images = train_images/255.0
test_images = test_images/255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Tensorflow keras API를 통해 모델에 대한 정의
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Adam Optimizer과 Cross Entropy Loss 선언
# 5 Epoch로 학습할 Data로 학습수행
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc*100)