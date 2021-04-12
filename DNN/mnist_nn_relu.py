import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist # fasion_mnis, cifar10, cifar100
from time import time
import os


"""
Input -> [Network] -> Output(ground-truth-output=loss)
       <-----------------loss를 미분한것(Gradient(기울기))을 
                         Backpropagation하면서 네트워크 학습
매우 작은(0에 가까운) Gradient가 한 두개면 문제없지만
네트워크가 딥하여 sigmoid function들이 여러 개일 경우
매우 작은 Gradient 값들이 많아져 곱해진다면 결국 이 Gradient가 소실되는,
즉 0에 가까워져서 네트워크가 전달받을 Gradient가 없게 되는 현상 발생
=> Vanishing Gradient
이런 현상이 생기면서 네트워크 학습이 잘 안되는 것 -> Sigmoid 함수의 문제점
"""

"""
- Relu
f(x) = max(0,x) -> 어떤 숫자 x를 받았을때, x>0 -> x, x<0 -> 0
Relu의 문제점은 0보다 작은 값일때 Gradient가 0

- leaky relu
x<0 -> x*a(0.01 or 0.1같은 매우 작은 수)
x>0 -> x
"""

# Checkpoint function
def load(model, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint = tf.train.Checkpoint(dnn=model)
        checkpoint.restore(save_path=os.path.join(checkpoint_dir,ckpt_name))

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
def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data() # 데이터를 불러옴
    # 채널을 하나 더 추가하는데, tensorflow가 인풋으로 받는 shape의 경우
    # [batch_size, height, width, channel]로 설정되어 있어야하기 때문에 추가하는 것
    # axis=-1로 되어있는데, -1의 의미는 채널을 어느 위치에 만들 것이냐를 결정
    # 즉 끝을 의미하는 -1를 넣어준 것.
    train_data = np.expand_dims(train_data, axis=-1) # [N, 28, 28] -> [N, 28, 28, 1]
    test_data = np.expand_dims(test_data, axis=-1)  # [N, 28, 28] -> [N, 28, 28, 1]
    train_data, test_data = normalize(train_data, test_data) # [0~255] -> [0~1]

    # 10 같은 경우 우리가 이용하는 이 데이터 셋의 라벨이 총 몇 개인지
    # 그것을 생각하고 값을 넣으면 된다 => One hot incoding
    # One hot incoding이란,
    # 예를 들어 7이라는 숫자라면 one hot incoding하지 않을 시 문자그대로 7이라고 표기된다
    # one hot incoding을 하면 10개의 개열을 갖고 있는 상태에서
    # 0000000100 7번째 위치하는 곳에 1이라고 적는고 나머지는 0으로
    # 이것을 컴퓨터가 해석하여 7이라는 것을 알게된다.
    train_labels = to_categorical(train_labels, 10) # [N,] -> [N, 10]
    test_labels = to_categorical(test_labels, 10) # [N,] -> [N,10]
    return train_data, train_labels, test_data, test_labels

def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32) / 255.0 # 255로 나누어 0~1로 나타나게함
    test_data = test_data.astype(np.float32) / 255.0
    return train_data, test_data


# Performance function
def loss_fn(model, images, labels):
    # model에 images를 넣어서 이 images의 숫자가 뭔지
    # logits 값을 아웃풋으로 추출
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=logits,
                                                                   y_true=labels,
                                                                   from_logits=True))
    # loss
    # 만약 이미지가 7이라면
    # [0][0][0][0][0][0][1][0][0] = label
    # [0.1][0.1][0.0][0.2][0.0][0.0][0.0][0.6][0.0][0.0] = softmax(logit)
    return loss

def accuracy_fn(model, images, labels):
    logits = model(images, training=False)
    # argmax : 이 logits과 이 labels에서 가장 큰 숫자의 위치가 어디를 구하는 것
    # argmax(batch_size, label_dim)
    # 위치를 보고 같은 위치인지 확인하여 True와 False를 반환할 것이다
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    # prediction을 type casting을 통해서 숫자 값으로 바꾼다
    # boolean 값보다는 숫자 값으로 해야 계산이 되기 때문이다.
    # reduce_mean : 평균
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
        # 이 loss에 해당하는 model들 어떤 weight들의 gradient를 리턴
    return tape.gradient(loss, model.variables)


# Model function
def flatten(): # shape을 펼쳐준다.
    return tf.keras.layers.Flatten()

def dense(labels_dim, weight_init): #
                            # units: 아웃풋으로 나가는 채널을 몇 개로 설정한 것인지를 의미
                            # bais : True이면 사용, Fasle면 노사용
                            # kernel_initializer : weight initializer
    return tf.keras.layers.Dense(units=labels_dim, use_bias=True, kernel_initializer=weight_init)

def relu(): # relu 사용
    return tf.keras.layers.Activation(tf.keras.activations.relu)


# Create model(class version)
class create_model_class(tf.keras.Model): # 클래스로 만들때는 tf.keras.Model을 사용해야한다.
    def __init__(self, label_dim): # label_dim : 몇 개의 아웃풋을 알려줘야하기 때문에 파라미터로 받음
        super(create_model_class, self).__init__()
        # weight의 초기값을 평균이 0 분산이 1인 가우시안 분포로 랜덤하게 설정
        weight_init = tf.keras.initializers.RandomNormal()

        # Sequential : 리스트 자료구조 타입
        self.model = tf.keras.Sequential()
        # flatten시키는 이유는 이후에 fully connected layer를 이용할 것이기 때문에 펴주는 것
        # convolution을 이용한다면 flatten 과정은 필요 없음
        self.model.add(flatten()) # [N, 28, 28, 1] -> [N, 784]

        for i in range(2):
            # [N,784] -> [N,256] -> [N,256]
            # 측 채널을 256으로 하고 relu activation 사용
            self.model.add(dense(256, weight_init))
            self.model.add(relu())
        self.model.add(dense(label_dim, weight_init)) # [N, 256] -> [N,10]
    def call(self, x, training=None, mask=None):
        x = self.model(x)
        return x # 어떻게 아웃풋을 내놔야 하는지에 대한 call 함수

# Create model(function version)
def create_model_function(label_dim):
    weight_init = tf.keras.initializers.RandomNormal()

    model = tf.keras.Sequential()
    model.add(flatten())

    for i in range(2):
        model.add(dense(256, weight_init))
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
# shuffle : 데이터 셋을 섞음, buffer_size : 인풋으로 들어가는 데이터들보다 숫자를 크게 해주면 된다.
# prefetch : 네트워크가 어떤 batch size만큼 학슴하고 있을 때 미리 메모리에 이 batch size만큼 올려라, 즉 학습을 빠르게 해줌
# batch : 면번만큼, batch size 몇개만큼 네트워크에 던져줄 건지 정함
print(len(train_x))
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=batch_size).\
    batch(batch_size, drop_remainder=True)

print(len(test_x))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=len(test_x)).\
    batch(len(test_x))


# Defin model & optimizer & writer
""" Model """
network = create_model_function(label_dim)

""" Training """
optimizer =tf.keras.optimizers.Adam(learning_rate=learning_rate)

""" Writer """
checkpoint_dir = 'checkpoints'
logs_dir = 'logs'

model_dir = 'nn_relu'
checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
check_folder(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, model_dir)
logs_dir = os.path.join(logs_dir, model_dir)


# Restore checkpoint & start train or test phase
if train_flag:
    # checkpoint : 네트워크가 학습을 하다 중간에 끊겼을 때
    # 재학습을 하고싶을때가 있다.
    # 그때 우리가 학습하면서 변경되었던 weight들을 불려내는 역할
    # 또는 학습이 끝났을때 inference
    # 즉 테스트 이미지에 대해서 정확도가 몇인지 보고싶을때
    # 이를 이용하여 테스트를 바로 할 수 있게 한다.
    #즉 학습되어서 저장된 weight들을 다시 부르는데 도움을 주는 것이라고 보시면 됩니다
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
    with summary_writer.as_default(): # for tensorboard
        for epoch in range(start_epoch, training_epochs):
            for idx, (train_input, train_label) in enumerate(train_dataset):
                grads = grad(network, train_input, train_label) # gradient를 구함
                optimizer.apply_gradients(grads_and_vars=zip(grads, network.variables))
                train_loss = loss_fn(network, train_input, train_label)
                train_accuracy = accuracy_fn(network, train_input, train_label)

                for test_input, test_label in test_dataset:
                    test_accuracy = accuracy_fn(network, test_input, test_label)

                tf.summary.scalar(name='train_loss', data=train_loss, step=counter)
                tf.summary.scalar(name='train_accuracy', data=train_accuracy, step=counter)
                tf.summary.scalar(name='test_accuracy', data=test_accuracy, step=counter)

                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_loss: %.8f, train_accuracy: %.4f, test_Accuracy: %.4f"
                      % (epoch, idx, training_iterations, time()-start_time, train_loss, train_accuracy, test_accuracy))
                counter += 1
        # weight 저장
        checkpoint.save(file_prefix=checkpoint_prefix + '-{}'.format(counter))

# test phase
else:
    _, _ = load(network, checkpoint_dir)
    for test_input, test_label in test_dataset:
        test_accuracy = accuracy_fn(network, test_input, test_label)
    print("test_Accuracy: %.4f" % (test_accuracy))