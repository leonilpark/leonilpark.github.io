# Library import


```python
# library import
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.preprocessing import minmax_scale
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
```

# Data import


```python
#load data to use Keras.dataset.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Format transformation
samples_num = X_train.shape[0]  
width = X_train.shape[1]
height = X_train.shape[2]
X_train = X_train.reshape(samples_num, width * height)
num_of_test_samples = X_test.shape[0]
X_test = X_test.reshape(num_of_test_samples, width * height)

# Convert int to float
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)


# Regulization
X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0)
X_test = minmax_scale(X_test, feature_range=(0, 1), axis=0)

# Data Division
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

# MLP


```python
# Multilayer Perceptron (MLP) Create
model = Sequential()

# Input-Layer
model.add(Dense(256, input_dim=width * height,  activation='elu'))
model.add(Dropout(0.3))

# 2nd-Layer
model.add(Dense(256,  activation='elu'))
model.add(Dropout(0.3))

# 3rd-Layer
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.3))

# 4th-Layer
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.3))

# 5th-Layer
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.3))


# 6th-Layer
number_of_class = 10 
model.add(Dense(number_of_class, activation='softmax'))  

# Cost Function & Optimizer Setting
# CE & Adam
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

# Traing & Evaluation


```python
# model training
training_epochs = 100
batch_size = 200
model.fit(X_train, y_train,batch_size,training_epochs)
```

    Epoch 1/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0260 - accuracy: 0.9931
    Epoch 2/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0280 - accuracy: 0.9924
    Epoch 3/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0308 - accuracy: 0.9918
    Epoch 4/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0303 - accuracy: 0.9919
    Epoch 5/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0275 - accuracy: 0.9923
    Epoch 6/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0281 - accuracy: 0.9925
    Epoch 7/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0298 - accuracy: 0.9927
    Epoch 8/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0257 - accuracy: 0.9935
    Epoch 9/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0244 - accuracy: 0.9936
    Epoch 10/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0271 - accuracy: 0.9933
    Epoch 11/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0265 - accuracy: 0.9930
    Epoch 12/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0270 - accuracy: 0.9930
    Epoch 13/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0269 - accuracy: 0.9930
    Epoch 14/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0241 - accuracy: 0.9938
    Epoch 15/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0341 - accuracy: 0.9916
    Epoch 16/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0294 - accuracy: 0.9928
    Epoch 17/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0281 - accuracy: 0.9928
    Epoch 18/100
    300/300 [==============================] - 1s 4ms/step - loss: 0.0258 - accuracy: 0.9928
    Epoch 19/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0262 - accuracy: 0.9928
    Epoch 20/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0224 - accuracy: 0.9937
    Epoch 21/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0273 - accuracy: 0.9929
    Epoch 22/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0277 - accuracy: 0.9931
    Epoch 23/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0248 - accuracy: 0.9936
    Epoch 24/100
    300/300 [==============================] - 1s 5ms/step - loss: 0.0249 - accuracy: 0.9934
    Epoch 25/100
    300/300 [==============================] - 2s 5ms/step - loss: 0.0236 - accuracy: 0.9938
    Epoch 26/100
    300/300 [==============================] - 2s 5ms/step - loss: 0.0282 - accuracy: 0.9928
    Epoch 27/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0248 - accuracy: 0.9938
    Epoch 28/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0255 - accuracy: 0.9931
    Epoch 29/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0257 - accuracy: 0.9934
    Epoch 30/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0280 - accuracy: 0.9928
    Epoch 31/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0260 - accuracy: 0.9934
    Epoch 32/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0232 - accuracy: 0.9937
    Epoch 33/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0239 - accuracy: 0.9940
    Epoch 34/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0242 - accuracy: 0.9942
    Epoch 35/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0260 - accuracy: 0.9934
    Epoch 36/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0267 - accuracy: 0.9936
    Epoch 37/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0223 - accuracy: 0.9947
    Epoch 38/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0321 - accuracy: 0.9929
    Epoch 39/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0230 - accuracy: 0.9941
    Epoch 40/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0240 - accuracy: 0.9941
    Epoch 41/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0246 - accuracy: 0.9936
    Epoch 42/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0260 - accuracy: 0.9936
    Epoch 43/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0356 - accuracy: 0.9915
    Epoch 44/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0253 - accuracy: 0.9934
    Epoch 45/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0254 - accuracy: 0.9934
    Epoch 46/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0197 - accuracy: 0.9944
    Epoch 47/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0184 - accuracy: 0.9947
    Epoch 48/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0211 - accuracy: 0.9947
    Epoch 49/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0256 - accuracy: 0.9940
    Epoch 50/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0276 - accuracy: 0.9932
    Epoch 51/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0241 - accuracy: 0.9941
    Epoch 52/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0241 - accuracy: 0.9941
    Epoch 53/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0223 - accuracy: 0.9945
    Epoch 54/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0217 - accuracy: 0.9941
    Epoch 55/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0241 - accuracy: 0.9941
    Epoch 56/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0221 - accuracy: 0.9945
    Epoch 57/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0238 - accuracy: 0.9943
    Epoch 58/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0277 - accuracy: 0.9937
    Epoch 59/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0203 - accuracy: 0.9951
    Epoch 60/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0230 - accuracy: 0.9942
    Epoch 61/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0237 - accuracy: 0.9946
    Epoch 62/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0249 - accuracy: 0.9941
    Epoch 63/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0196 - accuracy: 0.9949
    Epoch 64/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0295 - accuracy: 0.9933
    Epoch 65/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0303 - accuracy: 0.9935
    Epoch 66/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0241 - accuracy: 0.9944
    Epoch 67/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0376 - accuracy: 0.9926
    Epoch 68/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0334 - accuracy: 0.9924
    Epoch 69/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0250 - accuracy: 0.9937
    Epoch 70/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0227 - accuracy: 0.9951
    Epoch 71/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0211 - accuracy: 0.9947
    Epoch 72/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0215 - accuracy: 0.9944
    Epoch 73/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0232 - accuracy: 0.9946
    Epoch 74/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0281 - accuracy: 0.9934
    Epoch 75/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0228 - accuracy: 0.9946
    Epoch 76/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0218 - accuracy: 0.9952
    Epoch 77/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0176 - accuracy: 0.9952
    Epoch 78/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0190 - accuracy: 0.9953
    Epoch 79/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0215 - accuracy: 0.9953
    Epoch 80/100
    300/300 [==============================] - 2s 6ms/step - loss: 0.0257 - accuracy: 0.9939
    Epoch 81/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0222 - accuracy: 0.9944
    Epoch 82/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0224 - accuracy: 0.9945
    Epoch 83/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0187 - accuracy: 0.9952
    Epoch 84/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0202 - accuracy: 0.9944
    Epoch 85/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0231 - accuracy: 0.9952
    Epoch 86/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0239 - accuracy: 0.9948
    Epoch 87/100
    300/300 [==============================] - 2s 8ms/step - loss: 0.0317 - accuracy: 0.9941
    Epoch 88/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0387 - accuracy: 0.9923
    Epoch 89/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0303 - accuracy: 0.9930
    Epoch 90/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0648 - accuracy: 0.9925
    Epoch 91/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0296 - accuracy: 0.9925
    Epoch 92/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0467 - accuracy: 0.9919
    Epoch 93/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0268 - accuracy: 0.9932
    Epoch 94/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0238 - accuracy: 0.9940
    Epoch 95/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0212 - accuracy: 0.9945
    Epoch 96/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0450 - accuracy: 0.9934
    Epoch 97/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0290 - accuracy: 0.9934
    Epoch 98/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0357 - accuracy: 0.9920
    Epoch 99/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0216 - accuracy: 0.9944
    Epoch 100/100
    300/300 [==============================] - 2s 7ms/step - loss: 0.0232 - accuracy: 0.9945





    <tensorflow.python.keras.callbacks.History at 0x7faa2a454e50>




```python
# Model evaluation using test set
print('Model evaluation')
evaluation = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Accuracy: ' + str(evaluation[1]))
```

    Model evaluation
    50/50 [==============================] - 0s 2ms/step - loss: 0.1513 - accuracy: 0.9850
    Accuracy: 0.9850000143051147

