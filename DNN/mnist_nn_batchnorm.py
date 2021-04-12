import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from time import time
import os

"""
예를 들어 고양이 이미지를 확인하는 네트워크이다.
인풋으로 들어온 어떤 데이터의 분포 distribution이 생겨났는데
네트워크를 지나가면서(레이어들을 몇개 지나가면서)
이 distribution이 계속 망가지는 현상이 일어나게 된다.
결국 마지막에 받은 이미지가 어떤 이미지인지 알아보려고
네트워크는 맞춰야하는데 분포가 이상하게 바뀌어 학습이 조금 덜 되게 된다.
이러한 현상을 Internal Coveriate Shift라고 한다.
즉 Batch Normalization은 이러한 Internal Covariate Shift를 막기위한 것
-> 인풋으로 들어오는 distribution을 계속 Normalization을 시켜줘서
이 distribution을 항상 일정하게 만들어 보자는 뜻
즉 어떠한 x가 인풋으로 들어오고, 그 x에 대해서 어떤 Batch들의 평균
그리고 Batch들의 분산들을 이용해서 이렇게 Normalization 시켜주는 것
이후 이 x bar에 대해서 어떤 감마와 베타라는 학습이 되는 파라미터들을
이용해서 x의 hat을 만들어냄, 이 x hat을 레이어에 인풋으로 다시 추가
이러한 과정을 거치면 distribution이 일정하게 되어 네트워크가 학습을 진행한다.
"""

# Checkpoint function
def load(model, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt :
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint = tf.train.Checkpoint(dnn=model)
        checkpoint.restore(save_path=os.path.join(checkpoint_dir, ckpt_name))
        counter = int(ckpt_name.split('-')[1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# Data load & pre-processing function
def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]
    test_data = np.expand_dims(test_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10) # [N,] -> [N, 10]
    test_labels = to_categorical(test_labels, 10) # [N,] -> [N, 10]

    return train_data, train_labels, test_data, test_labels

def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    return train_data, test_data


# Performance function
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=logits, y_true=labels,
                                                                   from_logits=True))
    return loss

def accuracy_fn(model, images, labels):
    logits = model(images, training=False)
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.trainable_variables)


# Model function
def flatten() :
    return tf.keras.layers.Flatten()

def dense(label_dim, weight_init) :
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init)

def relu() :
    return tf.keras.layers.Activation(tf.keras.activations.relu)

### batch normalization
def batch_norm() :
    return tf.keras.layers.BatchNormalization()


# Create model (class version)
class create_model_class(tf.keras.Model):
    def __init__(self, label_dim):
        super(create_model_class, self).__init__()
        weight_init = tf.keras.initializers.glorot_uniform()
        self.model = tf.keras.Sequential()
        self.model.add(flatten())

        for i in range(4):
            self.model.add(dense(512, weight_init))
            ### batch normalization
            # 보통의 순서는 일반적으로 layer -> normalization -> activation이다
            # 위 순서 이외에도 normalization -> activation -> fully connected or convolution layer를 쓴다.
            self.model.add(batch_norm())
            self.model.add(relu())
        self.model.add(dense(label_dim, weight_init))

    def call(self, x, training=None, mask=None):
        x = self.model(x)
        return x

# Create model (function version)

def create_model_function(label_dim) :
    weight_init = tf.keras.initializers.glorot_uniform()
    model = tf.keras.Sequential()
    model.add(flatten())

    for i in range(4) :
        model.add(dense(512, weight_init))
        model.add(batch_norm())
        model.add(relu())
    model.add(dense(label_dim, weight_init))
    return model


# Define data & hyper-parameter
""" dataset """
train_x, train_y, test_x, test_y = load_mnist()

""" parameters """
learning_rate = 0.001
batch_size = 128

training_epochs = 1
training_iterations = len(train_x) // batch_size

label_dim = 10

train_flag = True

""" Graph Input using Dataset API """
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=batch_size).\
    batch(batch_size, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=len(test_x)).\
    batch(len(test_x))

# Define model & optimizer & writer
""" Model """
network = create_model_function(label_dim)

""" Training """
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

""" Writer """
checkpoint_dir = 'checkpoints'
logs_dir = 'logs'

model_dir = 'nn_batchnorm'

checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
check_folder(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, model_dir)
logs_dir = os.path.join(logs_dir, model_dir)


# Restore checkpoint & start train or test phase
if train_flag:

    checkpoint = tf.train.Checkpoint(dnn=network)

    # create writer for tensorboard
    summary_writer = tf.summary.create_file_writer(logdir=logs_dir)
    start_time = time()

    # restore check-point if it exits
    could_load, checkpoint_counter = load(network, checkpoint_dir)

    if could_load:
        start_epoch = (int)(checkpoint_counter / training_iterations)
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
    else:
        start_epoch = 0
        start_iteration = 0
        counter = 0
        print(" [!] Load failed...")

    # train phase
    with summary_writer.as_default():  # for tensorboard
        for epoch in range(start_epoch, training_epochs):
            for idx, (train_input, train_label) in enumerate(train_dataset):
                grads = grad(network, train_input, train_label)
                ### network.trainable_variables
                optimizer.apply_gradients(grads_and_vars=
                                          zip(grads, network.trainable_variables))

                train_loss = loss_fn(network, train_input, train_label)
                train_accuracy = accuracy_fn(network, train_input, train_label)

                for test_input, test_label in test_dataset:
                    test_accuracy = accuracy_fn(network, test_input, test_label)

                tf.summary.scalar(name='train_loss', data=train_loss, step=counter)
                tf.summary.scalar(name='train_accuracy', data=train_accuracy, step=counter)
                tf.summary.scalar(name='test_accuracy', data=test_accuracy, step=counter)

                print(
                    "Epoch: [%2d] [%5d/%5d] time: %4.4f, train_loss: %.8f, train_accuracy: %.4f, test_Accuracy: %.4f" \
                    % (epoch, idx, training_iterations, time() - start_time, train_loss, train_accuracy,
                       test_accuracy))
                counter += 1
        checkpoint.save(file_prefix=checkpoint_prefix + '-{}'.format(counter))

# test phase
else:
    _, _ = load(network, checkpoint_dir)
    for test_input, test_label in test_dataset:
        test_accuracy = accuracy_fn(network, test_input, test_label)

    print("test_Accuracy: %.4f" % (test_accuracy))
