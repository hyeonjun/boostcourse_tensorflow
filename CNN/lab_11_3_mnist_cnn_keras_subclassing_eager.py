# Importing Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

"""
NN Implementaion Flow in TensorFlow
1. Set hyper parameters - learin rate, training epochs, batch size, etc.
2. Make a data pipelining - use tf.data
3. Build a neural network model - use tf.keras.Model subclassing ***
4. Define a loss function - cross entropy
5. Calculate a gradient - use tf.GradientTape
6. Select an optimizer - Adam optimizer
7. Define a metric fro model's performance - accuracy
8. (optional) Make a checkpoint for saving
9. Train and Validata a neural network model
"""



# Hyper Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
tf.random.set_seed(777)

# Creating a Checkpoint Directory
cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'mnist_cnn_subclass'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

# MNIST/Fashion MNIST Data
## MNIST Dataset #################################################################
mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
## Fashion MNIST Dataset #########################################################
# mnist = keras.datasets.fashion_mnist
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##################################################################################

# Datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    buffer_size=100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

"""
Subclassing의 장점
: fully-customizable model을 만들 수 있다.
 => functional API를 사용하면 웬만한 복잡한 모델은 다 구현이 가능하다
    이렇게 subclassing 방법을 쓰면 그 안에다가 더 추가하고 싶은 operation같은 것들을 추가할 수 있는 여지가 많이 생긴다.
keras.Model의 클래스를 subclassing하는 방법은 가장 customizable한 모델을 만들 수 있는 방법이다.
방법
- 클래스 형태로 만들어 레이어 부분을 init 메서드에 다 선언한다.
- call 메서드에서 그 부분에 입력을 연결해주는 방식으로 구현된다.
"""

# Model Class
class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__() # 상위 메서드인 init 메서드를 호출
        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
        self.pool1 = keras.layers.MaxPool2D(padding='SAME')
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
        self.pool2 = keras.layers.MaxPool2D(padding='SAME')
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
        self.pool3 = keras.layers.MaxPool2D(padding='SAME')
        self.pool3_flat = keras.layers.Flatten()
        self.dense4 = keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.drop4 = keras.layers.Dropout(rate=0.4)
        self.dense5 = keras.layers.Dense(units=10)
    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net

model = MNISTModel()
temp_inputs = keras.Input(shape=(28,28,1))
model(temp_inputs)
model.summary()

# Loss Function
@tf.function
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        y_pred=logits, y_true=labels, from_logits=True
    ))
    return loss

# Calculating Gradient
@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)

# Calculating Model's Accuracy
@tf.function
def evaluate(model, images, labels):
    logits = model(images, training=False)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Creating a Checkpoint
checkpoint = tf.train.Checkpoint(cnn=model)

# Training
@tf.function
def train(model, images, labels):
    grads = grad(model, images, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
# train my model
print('Learing started. It takes sometime.')
for epoch in range(training_epochs):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0

    for images, labels in train_dataset:
        train(model, images, labels)
        # grads = grad(model, images, labels)
        # optimizer.apply_gradients(grads, model.variables))
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)
        avg_loss = avg_loss + loss
        avg_train_acc = avg_train_acc + acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step

    for images, labels in test_dataset:
        acc = evaluate(model, images, labels)
        avg_test_acc = avg_test_acc + acc
        test_step += 1
    avg_test_acc = avg_test_acc / test_step

    print('Epoch:', '{}'.format(epoch+1), 'loss = ', '{:.8f}'.format(avg_loss),
          'train accuracy = ', '{:.4f}'.format(avg_train_acc),
          'test accuracy = ', '{:.4f}'.format(avg_test_acc))
    checkpoint.save(file_prefix=checkpoint_prefix)
print('Learing Finished!')














