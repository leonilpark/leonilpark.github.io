# Optimizer & Training (TF 2.0 Expert)

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import datasets 
```

## Build Model

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

## Preporcess

Expert한 방법

- tf.data 사용

```python
mnist = tf.keras.datasets.mnist

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 차원 추가
x_train = x_train[..., tf.newaxis]
x_test = x_train[..., tf.newaxis]

# Data Normalization
x_train, x_test = x_train/255., x_test/255.
```

- from_tensor_slices()
- shuffle()
- batch()

## tf.data

```python
train_ds = tf.data.Datasets.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000) #1000은 버퍼 사이즈. 1000이 가장 적절함. 디폴드 값이라 생각
train_ds = train_ds.batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(32)
```

## Visualize Data

- matplotlib을 통해 데이터를 시각화 하기

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

- train_ds.take()

```python
#take를 통해 제한을 걸 것임
for image, label in train_ds.take(2):
	plt.title(label[0]) #배치 중 첫번째 것을 조회할 것임 그래서 0
	plt.imshow(image[0,:,:,0],'gray')
	plt.show()

#하나만 보고 싶을 경우
image, label = next(iter(train_ds))
for image, label in train_ds.take(2):
	plt.title(label[0]) #배치 중 첫번째 것을 조회할 것임 그래서 0
	plt.imshow(image[0, :, :, 0],'gray')
	plt.show()
```

## Training (Keras)

Keras로 학습 할 때는 기존과 같지만, train_ds는 generator라서 그대로 넣을 수 있음

```python
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy')
model.fit(train_ds, epochs=10000)
# train_ds를 사용하기 때문에 이미지와 레이블을 자동으로 나오고 배치사이즈도 지정해둔 상황
# 그래서 에폭수만 설정해주면 됌
```

## Optimization

- Loss Function
- Optimizer

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()
```

- Loss Function을 담을 곳
- Metrics

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
#평균값을 할 경우 그래프가 자연스럽게 그려지는 것을 볼 수 있음
train_accuracy = tf.keras.losses.SparseCategoricalCrossentropy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
#평균값을 할 경우 그래프가 자연스럽게 그려지는 것을 볼 수 있음
test_accuracy = tf.keras.losses.SparseCategoricalCrossentropy(name = 'test_accuracy')

```

## Training

@tf.function - 기존 session에서 열었던 것처럼 바로 작동 안하고, 그래프만 만들고 학습이 시작되면 돌아가도록 함.

```python
@tf.function

def train_step(images, labels):
	with tf.GradientTape() as tape : #Tape는 Gradient값을 얻으면서 학습이 되도록 함
		predictions = model(images)
		loss = loss_object(labels, predictions)
	
	gradients = tape.gradient(loss, model.trainable_variables)
	#trainable_varable을 하여금 loss를 통해서 기울기를 얻어 optimizer에서 적용을 시킬 것임
	optimizer.apply_gradients(zip(graidents,model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)
```

```python
@tf.function
def test_step(images, labels):
	predictions = model(images)
	t_loss = loss_object(labels, predictions)
	
test_loss(loss)
test_accuracy(labels, predictions)
```

```python
#반복적으로 작동할 수 있게 epoch 적용
for epoch in range(2):
	for images, labels in train_ds:
		train_step(images,labels)
	
	for t_images, t_labels in test_ds:
		test_step(test_images, test_labels)

	template = ' Epoch {}, loss : {}, Accauracy : {}, Test Loss :{}, Test Accauracy : {}'

	print(template.format(epoch+1,
												train_loss.result(),
												train_accuracy.result()*100,
												test_loss.result(),
												test_accuracy.result()*100))
```