import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

"""
NN Implementaion Flow in TensorFlow
1. Set hyper parameters - learin rate, training aepochs, batch size, etc.
2. Make a data pipelining - use tf.data
3. Build a neural network model - use tf.keras sequential APIs ***
4. Define a loss function - cross entropy
5. Calculate a gradient - use tf.GradientTape
6. Select an optimizer - Adam optimizer
7. Define a metric fro model's performance - accuracy
8. (optional) Make a checkpoint for saving
9. Train and Validata a neural network model
"""

# 1. Set hyper parameters
# Hyper Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Creating a Checkpoing Directory
cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'mnist_cnn_seq'

checkpoint_dir = os.path.join(cur_dir,ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

# 2. Make a data pipelining
# MNIST/Fashion MNIST Data
## MNIST Dataset ########################################################
mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#########################################################################

## Fashion MNIST Dataset ################################################
#mnist = keras.datasets.fashion_mnist
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##########################################################################

# Datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# (60000, 28,28) (60000,)      (10000,28,28) (10000,)
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.
train_images = np.expand_dims(train_images, axis=-1) # 4차원으로 만들기위해서
test_images = np.expand_dims(test_images, axis=-1)

train_labels = to_categorical(train_labels, 10) # one hot encoding
test_labels = to_categorical(test_labels, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    buffer_size=100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# 3. Build a neural network model
# Model Function
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                                  activation=tf.nn.relu, padding='SAME',
                                  input_shape=(28,28,1)))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                                  activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                                  activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))

    # fully connect layer
    model.add(keras.layers.Flatten()) # fully connect 하기위해 펴줌
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10))
    return model

model = create_model()
# model.summary()

# 4. Define a loss function - cross entropy
# Loss Function
@tf.function
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        y_pred=logits, y_true=labels, from_logits=True
    ))
    return loss

# 5. Calculate a gradient - use tf.GradientTape
# Calculating Gradient
@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)


# 7. Define a metric fro model's performance - accuracy
# Calculating Model's Accuracy
@tf.function
def evaluate(model, images, labels):
    logits = model(images, training=False)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# 6. Select an optimizer - Adam optimizer
# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 8. (optional) Make a checkpoint for saving
# Creating a Checkpoint
checkpoint = tf.train.Checkpoint(cnn=model)


# Training
@tf.function
def train(model, images, labels):
    grads = grad(model, images, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 9. Train and Validata a neural network model
# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0

    for images, labels in train_dataset:
        train(model, images, labels)
        # grads = grad(model, images, labels)
        # optimizer.apply_gradients(zip(grads, model.variables))
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

    print('Epoch:', '{}'.format(epoch+1), 'loss =', '{:.8f}'.format(avg_loss),
          'trian accuracy =', '{:.4f}'.format(avg_train_acc),
          'test accuracy =', '{:.4f}'.format(avg_test_acc))

    checkpoint.save(file_prefix=checkpoint_prefix)
print('Learing Finished!')