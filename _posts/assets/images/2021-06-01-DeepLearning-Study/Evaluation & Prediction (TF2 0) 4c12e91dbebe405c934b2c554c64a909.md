# Evaluation & Prediction (TF2.0)

- 이전 expert 이후로 계속 작성하시기 바랍니다.

## Evaluating

- 학습한 모델 확인

```python
model.evaluate(test_x, test_y, batch_size = batch_size)
```

### 결과 확인

input으로 들어갈 이미지 데이터 확인

```python
import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
```

```python
test_image = test_x[0, :, :, 0]
test_image.shape
```

```python
plt.title(test_y[0])
plt.imshow(test_image,'gray')
plt.show()
```

- 모델에 input Data로 확인 할 이미지 데이터 넣기

```python
pred = model.predict(test_image.reshape(1, 28, 28, 1))
```

```python
pred.shape #크기가 크지 않으니 바로 확인
pred #softmax를 사용했으니 가장 높은 수치를 얻은 로드가 정답임.
```

- np.argmax

```python
np.argmax(pred) #7번째에 있는 인덱스의 수치가 가장 높다는 것을 확인할 수 있음.
```

## Test Batch

Batch로 Test Dataset 넣기

```python
test_batych = test_x[:32]
```

Batch Test Dataset을 모델에 넣기

```python
preds = model.predict(test_batch)
preds.shape
```

결과 확인

```python
plt.imshow(test_batch[1, : , :, 0],'gray')
plt.show()
```