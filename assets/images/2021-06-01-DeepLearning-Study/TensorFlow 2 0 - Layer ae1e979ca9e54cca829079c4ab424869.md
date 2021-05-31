# TensorFlow 2.0 - Layer

## Layer Explaination

```python
import tensorflow as tf
```

### Input Image

- Input으로 들어갈 DataSet을 들여다보고 시각화까지

 

- 패키지 로드
- os
- glob
- matplotlib

```python
import matplotlib.pyplot as plt
%matplotlib online

from tensorflow.keras import datasets

(train_x, train_y), (test_x,test_y) = datasets.mnist.load_data()
image = train_x[0]
image.shape #확인해주는 과정이 필요함
plt.imshow(image, 'gray')
plt.show()

image = image[tf.newaxis, ..., tf.newaxis] #딥러닝하기 위해서 변환해줘야한다. 이 전 강의 참고
```

### Feature Extraction

![TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled.png](TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled.png)

- 이미지를 들어갔을 때, 바로 예측을 하는 것이 아니라 Feature extraction을 거친 후 예측이 들어간다.

### Convolution Layer

![TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%201.png](TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%201.png)

- filters : layer에서 나갈 때 몇개의 filter를 만들 것인지 (다른 말로는 weihts, filters,
- kernel_size : filter(Weight)의 사이즈(대부분 (3,3))
- strides : 몇 개의 pixel을 skip 하면서 훑어지나갈 것인지 (사이즈에도 영향을 준다.)
- padding : zero padding을 만들 것인지. VALID는 Padding이 없고, SAME은 Padding이 있다. (사이즈에도 영향을 준다.)
- activation : Activation Function을 만들 것인지. 당장 설정 안해도 Layer층을 따로 만들 수 있다.

```python
tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding = 'SAME', activation='relu')
```

→ 이때 다음과 같이도 쓸 수 있다. (정방행렬 생각)

```python
tf.keras.layers.Conv2D(3,3,1, padding = 'SAME', activation='relu')
```

### Visualization

- tf.keras.layers.Conv2D

```python
image = tf.cast(image, dtype=tf.float32)
layer = tf.keras.layers.Conv2D(3,3,1,padding='SAME')
output = layer(image)

#이미지 시각화해서 비교하기
plt.subplot(1,2,1)
plt.imshow(image[0, :, :, 0],'gray')
plt.subplot(1,2,2)
plt.imshow(output[0, :, :, 0],'gray')
plt.show()
```

- weight 불러오기
    - layer.get_weights()

    ```python
    weight = layer.get_weights() #output은 리스트형
    #len(weight)를 보면 2인데, 첫번째는 모양이고, 두번째는 bias이다.

    plt.figure(figsize=(15,5)) #히스토그램 조회
    plt.subplot(131)
    plt.hist(output.numpy().ravel(),range=[-2,2])
    plt.ylim(0,200)

    plt.subplot(132) #필터
    plt.title(weight[0].shape)
    plt.imshow(weight[0][:,:,0,0],'gray')

    plt.subplot(133) #변한 것
    plt.title(output.shape)
    plt.imshow(output[0,:,:,0],'gray')
    plt.colorbar()
    plt.show()
    ```

    ### Activation Function

    ![DeepLearning%20%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%8B%E1%85%A5%2086ce4bada0e648a4bbba52a6d09e2470/Untitled%205.png](DeepLearning%20%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%8B%E1%85%A5%2086ce4bada0e648a4bbba52a6d09e2470/Untitled%205.png)

    - 텐서값을 어떻게 조절할지에 대한 함수
    - ReLU의 경우 0미만인 것들은 다 0으로 처리한다고 하는 것

    ```python
    import numpy as np

    tf.keras.layers.ReLU()

    actvation_layer = tf.keras.layers.ReLU()
    output = activation_layer(output)
    np.min(output),np,max(output) #값을 확인해서 수가 어떻게 변했는지 확인
    ```

    ### Pooling

    ![CNN%20Model%20%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%2028a0703f145643e6af760fd57b7ef182/Untitled%202.png](CNN%20Model%20%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9%2028a0703f145643e6af760fd57b7ef182/Untitled%202.png)

     

    - tf.keras.layers.MaxPool2D
    - 이미지를 받아서 반으로 줄여서 압축을 하는 과정

    ```python
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='SAME')
    pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='SAME')
    pool_output = pool_layer(output)

    #시각화
    plt.subplot(121)
    plt.title(pool_output.shape)
    plt.imshow(pool_output[0, :, :, 0],'gray')
    plt.colorbar()
    plt.show()
    ```

    ### Fully Connected

    ![TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%202.png](TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%202.png)

    → y = wX + b (w : weight, b : bias)

    ### Flatten

    ![TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%203.png](TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%203.png)

    - tf.keras.layers.Flatten()

    ```python
    import tensorflow as tf
    tf.keras.layers.Flatten()
    layer = tf.keras.layers.Flatten()
    flatten = layer(output)
    flatten.shape #28 * 28 * 5 -> [1,3920] 여기 1은 배치사이즈

    #시각화
    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.hist(flatten.numpy().ravel())

    plt.subplot(212)
    plt.imshow(flatten[:,:100])
    plt.show()
    #시각화하면 0에 대부분 몰려있어 잘 보이지 않음
    ```

    ### Dense

    ![TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%204.png](TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%204.png)

    - tf.keras.layers.Dense

    ```python
    tf.keras.layers.Dense(32, activation='relu')
    #32는 units갯수 : 3920을 받았는데 노드를 32개를 만들어서 연결하겠다는 뜻
    layer = tf.keras.layers.Dense(32,activation='relu')
    output = layer(flatten)
    output.shape #[1,32] 3920 -> 32로 줄어든 것을 볼 수 있음

    layer_2 = tf.keras.layers.Dense(10,activation='relu')
    output_example = layer_2(output)
    output_example #32개를 받아서 10개로 내보내는 것을 볼 수 있음
    ```

    ### Dropout (드롭아웃)

    ![TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%205.png](TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%205.png)

    - tf.keras.layers.Dropout

    ```python
    layer = tf.keras.layers.Dropout(0,7)
    output = layer(output)

    output.shape #shape 확인
    ```

    ### Build Model

    ![TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%206.png](TensorFlow%202%200%20-%20Layer%20ae1e979ca9e54cca829079c4ab424869/Untitled%206.png)

    - 모델 생성

    ```python
    from tensorflow.keras import layers

    input_shape = (28, 28, 1)
    num_classes = 10

    inputs = layers.Input(shape = input_shape)

    # Feature Extraction
    # 1st Block
    net = layers.Conv2D(32, 3, padding='SAME')(inputs)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(32, 3, padding='SAME')(net)
    net = layers.Activation('relu')(net)
    net = layers.MaxPool2D((2,2))(net)
    net = layers.Dropout(0.25)(net)

    #2nd Block
    net = layers.Conv2D(64, 3, padding='SAME')(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(64, 3, padding='SAME')(net)
    net = layers.Activation('relu')(net)
    net = layers.MaxPool2D((2,2))(net)
    net = layers.Dropout(0.25)(net)

    # Fully Connected
    net = layers.Flatten()(net)
    net = layers.Dense(512)(net)
    net = layers.Activation('relu')(net)
    net = layers.Dropout(0.25)(net)
    net = layers.Dense(num_classes)(net)
    #이때 num_classes는 데이터의 클래스 개수로 해야 된다.(노드의 개수)
    net = layers.Activation('softmax')(net)
    #softmax는 확률로 나타내는 것을 말한다.

    model = tf.keras.Model(inputs = inputs, outputs = net, name = 'CNN_Basic')
    ```

    - 모델 정보 얻기

    ```python
    model.summary()
    ```