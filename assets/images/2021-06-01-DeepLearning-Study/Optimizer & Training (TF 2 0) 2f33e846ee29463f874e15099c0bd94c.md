# Optimizer & Training (TF 2.0)

- 기본 패키지 import

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets
```

### 데이터셋 준비

```python
(train_x,train_y),(test_x,test_y) = datasets.mnist.load.data()
```

### 모델 빌딩

```python
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

## Optimization

→ 모델을 학습하기 전 설정해줘야 하는 항목

- Loss Function
- Optimization
- Metrics

## Loss Function

Loss Function 방법 확인

### - Categorical(클래스가 2개 이상일 때) & Binary(클래스가 2개일 때)

```python
loss = 'binary_crossentropy'
loss = 'categorical_crossentropy'
```

### - sparse_categorical_crossentropy & categorical_crossentropy

- One-Hot incording을 하지 않았을 경우

(One-Hot incordin을 사용하지 않을 것이기 떄문에 이를 Loss Function으로 사용할 것임.)

```python
loss = tf.kears.losses.sparse_categorical_crossentropy
```

- One-Hot incording을 했을 경우

```python
tf.keras.losses.categorical_crossentropy
```

- Binary를 주고 싶은 경우

```python
tf.keras.losses.binary_crossentropy
```

## Metrics

모델을 평가하는 방법

### Accuracy를 이름으로 넣는 방법

```python
metrics = [tf.keras.metrics.Accuracy()]
```

```python
tf.keras.metrics.Accuracy() #방법 1
tf.keras.metrics.Precision() #방법 2
tf.keras.metrics.Recall() #방법 3
```

## Complie

Optimizer 적용

- sgd
- rmsprop
- adam

```python
tf.keras.optimizers.SGD()
tf.keras.optimizers.RMSprop()
opt = tf.keras.optimizers.Adam() #아담을 사용하고자 함
```

컴파일

```python
model.compile(optimizer = opt,
							loss = loss,
							metrics=metrics, 'recall', 'precision')
```

## Prepare Dataset

학습에 사용할 데이터셋 준비

- shape 확인

```python
tarin_x.shape, train_y.shape
test_x.shape, test_y.shape
```

- 차원 수 늘리기

```python
import numpy as np
np.expand_dims(train_x, -1).shape #방법 1
tf.expand_dims(train_x, -1).shape #방법 2
train_x = train_x[..., tf.newaxis] #방법 3
test_x = train_x[..., tf.newaxis]
```

- Rescaling

```python
train_x = train_x/ 255.
test_x = test_x/ 255.
```

## Training

학습하기

- 학습용 하이퍼 파라메터 설정
    - num_epochs
    - batch_size

    ```python
    num_epochs = 1
    batch_size = 32
    ```

    - model.fit

    ```python
    model.fit(train_x,
    					train_y,
    					batch_size = batch_size,
    					shuffle=True,
    					epochs=num_epochs)

    # 셔플은 evaluation을 할 때는 사용하지 않는다.
    # bias가 걸리거나 오버피팅이 걸리기 때문에 학습에서는 해줘야 한다.

    # 셔플은 말 그대로 순서대로 학습을 진행한다는 것인데 끝에 가서는
    # 처음에 학습한 내용을 잊을 수 있기 때문에 해줘야 한다.
    # -> 즉 데이터를 섞어줘야 한다.
    ```

## Check History

학습 과정(History) 결과 확인

```python
hist.history
```