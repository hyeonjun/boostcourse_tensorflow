import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

"""
CNN은 image classification에서 가장 널리 사용되는 뉴럴 네트워크이다
이 CNN은 세 가지 종유의 레이어로 구성되어 있습니다.
Convolution layer, Pooling layer, Fully-connected layer
이 레이어들을 적당히 섞어서 배치
주로 convolution과 pooling을 앞쪽에 배치하고 Fully-connected layer를 마지막쪽에 배치하는 식으로 구성된다.
convolution layer와 Pooling layer에서는 feature extraction이라는 역할을 ㅎ가ㅗ
Fully-connected layer는 classification 역할을 합니다.
 - feature extraction
  : 어떤 특징을 뽑아 내는 것
 - Fully-connected layer에서는 이러한 특징들을 모아
   예를 들어 앞 부분이 뾰족, 주변에 파란색 물을 보고 이것이 배 라는 것을 
   실제 판단하는 역할을 한다.

이미지를 입력받는 경우가 많은데 이미지를 입력받을 때는
2D convolution layer를 많이 사용하게 된다

EX)
* 2D Convolution Layer
- Convolution Layer(항상 image를 받는 것이 아니고, 
    이전 Layer의 출력 feature may을 입력으로 받을 것이기 때문에
    그 경우에는 큰 숫자가 되는 경우가 일반적)
 32*32*3 image
 (32height 32width 3channel(이미지의 경우 RGB인 세 개의 채널을 갖기때문에 3, 흑백일 경우 채널은 1))
- Convolution filter
 5*5*3 filter
 (Convolution Layer에 입력으로 들어오는 것의 채널
 또는 feature map의 채널과 filter의 채널이 같아야함) 
- Feature map
  Activation maps
  => fiter를 하나만 쓰는 것이 아니라 여러 장의 filter를 사용할 것이기
  때문에 Feature map은 여러 장(사용한 filter의 수)이 되고 채널 수가 증가한다

2D Convolution Layer - Computation 
EX)
=> 1*1 + 1*0 + 1*1 + 0*0 + 1*1 + 1*0 + 0*1 + 0*0 + 1*1 = 4
1 1 1 0 0                         1 0 1           4 _ _
0 1 1 1 0                         0 1 0           _ _ _
0 0 1 1 1 을 가져와 convolution =>  1 0 1 filter => _ _ _ output feature map  
0 1 1 0 0

* 옵션
- stride : 한번의 Convolution 연산이 끝나고 옆으로 이동할 때
           몇 칸 옆으로 이동할 것인지에 대한 값
- Zero Padding
 padding을 하지 않고 convoluton 연산을 하면 이미지도 줄어들게 되는데
 이렇게 반복되면 레이어를 깊게 쌓는데 지장이 생깁니다.
 그것을 막기위해 주변에 0으로 padding을 해줍니다.
 
Convolution 연산을 하고 나면 Activation Function을 통과시키는데
Activation Function은 relu를 가장 많이 쓴다.
relu를 사용하면 음수는 0으로 바꿔주고 양수는 그대로 통과시킨다.
"""


"""
tf.keras.layers.Conv2D(
    filters : convolution filter의 수 => Output Filter map의 채널을 몇으로 할 것인지
    kernel_size : Convolution filter 크기(3*3=>3 or (3,3), [3,3] 등)
    strides : 숫자로 하나만 써도 되고 tuple이나 list로 두가지를 써도된다
    padding : valid or same
        valide는 패딩을 하지 않음
        same은 strides가 1인 경우를 기준으로 했을 때 입력과 출력의 사이즈가 똑같아지도록 해주는 것
    data_format : channels_last와 channels_first가 있음
        tensorflow에서는 기본적으로 channels_last가 default이다
        channels_last의 경우 batch(mini-batch size), height, width, channels 순이고
        channels_first의 경우 batch(mini-batch size), channels, height, width 순이다
    activation : Activation function을 넣어주는데 relu같은 것을 쓰면 된다
    use_bias : bias를 쓸 것인지에 대한 부분
    kernel_initializer
    bias_initializer => 이 Convolution filter와 bias에 initailizing 할 때
                        어떻게 해줄 것인지(어떤 방법을 쓸건지)에 대한 부분
    kernel_regularizer
    bias_regularizer => L2_regularization 같은 방법들을 기술
    kernel dimension : height, width, in_channel, out_channel                
)
"""


"""
image = tf.constant([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]], dtype=np.float32)
# print(image.shape)
plt.imshow(image.numpy().reshape(3,3), cmap='Greys')
"""

"""
print(" 1 filter --------------------------------------------------")
print("image.shape", image.shape)
#         (batch, h, w, channels)
# image.shape (1, 3, 3, 1)
weight = np.array([[[[1.]],[[1.]]],
                   [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
# weight.shape (2, 2, 1, 1)
weight_init = tf.constant_initializer(weight)
# conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID',
#                              kernel_initializer=weight_init)(image)
conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='SAME',
                             kernel_initializer=weight_init)(image)

print("conv2d.shape", conv2d.shape)

# padding='VALID' 일때
# conv2d.shape (1, 2, 2, 1)
# print(conv2d.numpy().reshape(2,2))
# [[12. 16.]
#  [24. 28.]]
# plt.imshow(conv2d.numpy().reshape(2,2), cmap='gray')

# padding='SAME' 일때
# conv2d.shape (1, 3, 3, 1)
# print(conv2d.numpy().reshape(3,3))
# [[12. 16.  9.]
#  [24. 28. 15.]
#  [15. 17.  9.]]
# plt.imshow(conv2d.numpy().reshape(3,3), cmap='gray')
"""

"""
print(" 3 filter --------------------------------------------------")
print("iamge.shape", image.shape)
# iamge.shape (1, 3, 3, 1)
weight = np.array([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                   [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
# weight => 1 1    10 10    -1 -1 
#           1 1    10 10    -1 -1

print("weight.shape", weight.shape)
#          (filter h, w, filter channel)
# weight.shape (2, 2, 1, 3)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding='SAME',
                             kernel_initializer=weight_init)(image)
print("conv2d.shape", conv2d.shape)
#          (batch, h, w, channel)
# conv2d.shape (1, 3, 3, 3)
feature_maps = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    print(feature_map.reshape(3,3))
    #1 [[12. 16.  9.]   2 [[120. 160.  90.]    3 [[-12. -16.  -9.]
    #   [24. 28. 15.]      [240. 280. 150.]       [-24. -28. -15.]
    #  [15. 17.9.]]        [150. 170.  90.]]      [-15. -17.  -9.]]
    plt.subplot(1,3,i+1),plt.imshow(feature_map.reshape(3,3), cmap='gray')
plt.show()
"""

# =============================================================================
"""
Pooling Layer에서 사용하는 연산은 Max Pooling과 Average Pooling이 있음
Pooling 연산은 채널별로 따로 연산하게 된다. 그래서 Output Feature의 채널이 input 채널과 같아지는 것

* Max Pooling(SubSampling) : filter당 가장 큰 값을 가져옴
1 1 2 4                       1 1 5 6 => 6
5 6 7 8                       2 4 7 8 => 8
3 2 1 0                       3 2 1 2 => 3       6 8
1 2 3 4 일때 stride 2이다. =>   1 0 3 4 => 4  =>   3 4

* Average Pooling : filter 당 평균을 구하면 된다.
1 1 2 4                       1 1 5 6 => 13/4 = 3.25
5 6 7 8                       2 4 7 8 => 21/4 = 5.25
3 2 1 0                       3 2 1 2 => 8/4  = 2        3.25 5.25
1 2 3 4 일 때 stride 2이다. =>  1 0 3 4 => 8/4  = 2  =>      2   2

feature extraction 할때 Max Pooling을 많이 하는 경향이 있음
이유는 Convolution 연산 결과로 나온 feature map이라고 가정했었는데
위 입력 16개 숫자가 Convolution 연산 결과로 큰 숫자가 나왔다는 것은 Convolution filter가
찾아내고자 했던 특징에 더 가까운 값이라는 뜻이고 특징에 더 가까울 수록
큰 숫자를 연산 결과로 나오게 된다. 그래서 가장 특징에 가까운 값을 가져오기 위해
Max Pooling을 하는 것이다.
"""
# print(" MAX POOLING --------------------------------------------------")
# MAX POOLING
"""
tf.keras.layers.MaxPool2D(
    pool_size : pooling할 때의 filter size, ex) 2 or(=) (2,2)
    strides
    padding : valid or same
    data_format : channels_last(default) or channels_first
)
"""
# image = tf.constant([[[[4],[3]],
#                       [[2],[1]]]], dtype=np.float32)
# pool = keras.layers.MaxPool2D(pool_size=(2,2), strides=1, padding='VALID')(image)
# print(pool.shape) # (1, 1, 1, 1)
# print(pool.numpy()) # [[[[4.]]]]


# =============================================================================
"""
print(" SAME:Zero paddings --------------------------------------------------")
image = tf.constant([[[[4],[3]],
                      [[2],[1]]]], dtype=np.float32)
pool = keras.layers.MaxPool2D(pool_size=(2,2), strides=1, padding='SAME')(image)
print(pool.shape) # (1, 2, 2, 1)
print(pool.numpy())
# [[[[4.]
#    [3.]]
#
#   [[2.]
#    [1.]]]]
"""

# =============================================================================

mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#mnist = keras.datasets.fashion_mnist
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels),(test_images, test_labels)= mnist.load_data()
train_images = train_images.astype(np.float32)/255. # 255로 나누어 0~1사이 값으로 나오도록 scaling
test_images = test_images.astype(np.float32)/255.
img = train_images[0] # 5라고 써져있는 손글씨를 가져오게됨
# plt.imshow(img, cmap='gray')
# plt.show()

#               -1이라고 하면 알아서 이 값을 채움
#               (batch, w, h, channel)
img = img.reshape(-1,28,28,1) # Convolution 연산에 집어넣기 위해 4차원으로 바꿔줌
img = tf.convert_to_tensor(img) # 이 데이터 셋은 numpy ndarray 타입이기 때문에 API에 넣으려면 tensor로 바꿔야함
# 그래서 tf.convert_to_tensor를 사용하여 tensor로 변경함

weight_init = keras.initializers.RandomNormal(stddev=0.01)
#                       filter 개수는 5개 크기는 3*3
conv2d = keras.layers.Conv2D(filters=5, kernel_size=3, strides=(2,2),
                             padding='SAME', kernel_initializer=weight_init)(img)
# 28*28 이미지가 14*14로 바뀜

print(conv2d.shape) # (1, 14, 14, 5)
# feature_maps = np.swapaxes(conv2d, 0, 3)
# for i, feature_map in enumerate(feature_maps):
#     plt.subplot(1,5,i+1), plt.imshow(feature_map.reshape(14,14), cmap='gray')
# plt.show()

# pooling ----------------------------------
#                               2*2 filter사용                    MAX Pooling
pool = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')(conv2d)
print(pool.shape) # (1, 7, 7, 5)
feature_maps = np.swapaxes(pool,0,3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1,5,i+1), plt.imshow(feature_map.reshape(7,7), cmap='gray')
plt.show() # feature map을 볼 수 있음

