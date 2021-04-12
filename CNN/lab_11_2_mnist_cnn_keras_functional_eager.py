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
3. Build a neural network model - use tf.keras functional APIs ***
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
model_dir_name = 'mnist_cnn_func'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)


# MNIST/Fashion MNIST Data

## MNIST Dataset #########################################################
mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
##########################################################################

## Fashion MNIST Dataset #################################################
#mnist = keras.datasets.fashion_mnist
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##########################################################################


# Datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32)/255.
test_images = test_images.astype(np.float32)/255.
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
    buffer_size=100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

# Model Function
"""
Limitation of Sequential API
 - Multi-input models
 - Multi-output models
 - Models with shared layers(the same layer called several times)
 - Models with non-sequential data flow(e.g.. Residual connections)
 위 케이스(inception이나 residual block 같은)를 만들지 못함
Function API는 가능하게 해준다.
"""

"""
ex) Residual Block
   |-------256-d-->|
1*1,64             |
   |(relu)         |
3*3,64             |
   |(relu)         |
1*1,256            |
   |               |
   +  <------------|
   |(relu) 
   |
inputs = keras.Input(shape(28,28,256))
conv1 = keras.layers.Conv2D(filters=64, kernel_size=1, padding='SAME',
    activation=keras.layers.ReLU())(inputs)
conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME',
    activation=keras.layers.ReLU())(conv1)
conv3 = keras.layers.Conv2D(filters=256, kernel_size=1, padding='SAME')(conv2)
add3 = keras.layers.add([conv3,inputs])
relu3 = keras.layers.ReLU()(add3)
model = keras.Model(inputs=inputs, outputs=relu3)
"""
def create_model():
    inputs = keras.Input(shape=(28,28,1)) # input layer 생성
    # conv1에 inputs를 붙여서 입력이 inputs임을 명시해줌
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)(inputs)
    pool1 = keras.layers.MaxPool2D(padding='SAME')(conv1)
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)(pool1)
    pool2 = keras.layers.MaxPool2D(padding='SAME')(conv2)
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)(pool2)
    pool3 = keras.layers.MaxPool2D(padding='SAME')(conv3)

    # fully-connected layer
    pool3_flat = keras.layers.Flatten()(pool3)
    dense4 = keras.layers.Dense(units=256, activation=tf.nn.relu)(pool3_flat)
    drop4 = keras.layers.Dropout(rate=0.4)(dense4)
    logits = keras.layers.Dense(units=10)(drop4)
    return keras.Model(inputs=inputs, outputs=logits)

model = create_model()
model.summary()


# Loss Function
@tf.function
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        y_pred=logits, y_true=labels, from_logits=True
    ))
    return loss

# Calculate Gradient
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
    avg_trian_acc = 0.
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
        avg_trian_acc = avg_trian_acc + acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_trian_acc = avg_trian_acc / train_step

    for images, labels in test_dataset:
        acc = evaluate(model, images, labels)
        avg_test_acc = avg_test_acc + acc
        test_step += 1
    avg_test_acc = avg_test_acc / test_step

    print('Epoch:', '{}'.format(epoch+1), 'loss =','{:.8f}'.format(avg_loss),
          'train accuracy = ','{:.4f}'.format(avg_trian_acc),
          'test accuracy = ','{:.4f}'.format(avg_test_acc))

    checkpoint.save(file_prefix=checkpoint_prefix)

print('Learning Finished!')



















