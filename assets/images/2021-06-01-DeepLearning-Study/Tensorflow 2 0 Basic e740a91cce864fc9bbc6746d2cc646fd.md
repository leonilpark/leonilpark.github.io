# Tensorflow 2.0 Basic

## Tensor 생성

- List 생성

```python
[1, 2, 3]
```

### Array생성

- tuple이나 list 둘 다 np.array()로 만들어 array를 만들 수 있다.

```python
import numpy as np
np.array([1, 2, 3])
```

### Tensor 생성

- tf.constant()
    - list → Tensor

```python
import tensorflow as tf
tf.constant([1,2,3])
```

- list → Tensor

```python
tf.constant(((1,2,3),(1,2,3)))
```

- Array → Tensor

```python
arr = np.array([1,2,3])
tf.consatant(arr)
```

## Tensor의 정보 확인

- shape 확인

```python
tensor.shape
```

- data type 확인
    - 주의 : Tensor를 생성 할 때도 data type을 정해주지 않기 때문에 data type에 대한 혼동이 올 수 있다.
    - Data Type에 따라 모델의 무게나 성능 차이에도 영향을 줄 수 있음

```python
tensor.dtype
```

- data type 정의

```python
tf.constant([1,2,3],dtype=tf.float32)
```

- data type 변환

    → Numpy에서 astype()을 거쳤듯이, TensorFlow에서는 tf.cast를 사용합니다.

    ```python
    arr = np.array([1,2,3],dtype=np.float32)
    arr.astype(np.uint8)

    #윗 방식을 Numpy에서 사용했지만 tf.cast는 다음과 같이 사용합니다.
    ts = tf.constant([1,2,3],dtype=tf.float32)
    tf.cast(ts,dtype=tf.uint8)
    ```

- Tensor에서 Numpy 불러오기

    → .numpy()

    ```python
    ts.numpy()
    ```

- Tensor에서 Numpy 불러오기

    → np.array()

    ```python
    np.array(ts)
    ```

- type()를 사용하여 numpy array로 변환된 것을 확인

    ```python
    type(ts.numpy())
    ```

## 난수 생성

![Tensorflow%202%200%20Basic%20e740a91cce864fc9bbc6746d2cc646fd/Untitled.png](assets/images/Tensorflow-Basic/Untitled.png)

- Normal Distribution은 중심 극한 이론에 의해서 연속적인 모양
- Uniform Distibution은 불연속적이며 일정한 분포

- numpy에서는 normal distribution을 기본적으로 생성한다.
    - np.random.randn()

    ```python
    np.random.randn(10) #randn(n) n은 생성하고자 하는 숫자
    ```

- tf.random.normal
    - TensorFlow에서의 normal distribution

    ```python
    np.random.normal(10) #0부터 10까지의 임의의 숫자 추출

    tf.random.normal([3,3]) #3*3형태로 임의 추출
    ```

- tf.random.uniform
    - TensorFlow에서의 unifrom dtistribution

    ```python
    tf.random.uniform([3,3]) #3*3형태로 임의 추출
    ```
