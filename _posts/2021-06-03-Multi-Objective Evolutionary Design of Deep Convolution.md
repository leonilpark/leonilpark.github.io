---
title : Multi-Objective Evolutionary Design of Deep Convolutional Neural Networks for Image Classification 논문 리뷰
date : 2021-06-03 01:05:03 +0000
last_modified_at : 2021-06-03 01:10:03 +0000
---

# Multi-Objective Evolutionary Design of Deep Convolutional Neural Networks for Image Classification 논문 리뷰

[논문 링크](https://paperswithcode.com/paper/multi-criterion-evolutionary-design-of-deep)

CNN의 한계

- solely optimized for classficiation performance
- one deployement senario
- Search process requires vast compitational resources in most approaches

제안

- evlutionary algorithm for searching neural architectures under multiple objectives(classification performance , FLOPs)

한계의 보완

- 구성 요소를 계속 재결합하고 수정하는 유전적연산을 통해 파레토최적을 approximate하는 구조를 채워 넣어 한계를 보완
- 구조를 축소하여 베이지안 모델 학습을 통해 과거의 결과와 공유 된 패턴을 강화하여 효율성 증가

    → 위를 사용할 시 image classification에서 효율적인 설계가 가능함

### NAS

- Neural architecture search
- NAS를 이용하면 CNN 모델의 구조 최적화를 할 수 있다.
- 이는 설계를 최적화 문제로 간주해 프로세스를 완화시킬 수 있는 경로를 제시한다

### NSGANet

- Multi-objective 진화 알고리즘을 제시함
- 이 논문의 주요 알고리즘
- 구조 구성 요소를 재결합하고 수정하는 작업을 통해 한번의 실행으로 전체의 파레토최적을 시킬 수 있도록 아키텍처를 최적화 시킨다.
- 이때 구조를 축소하고 Bayesian Network based distribution estimation operator을 통해 과거의 결과를 구조 간의 공유된 패턴을 강화해 효율성 향상.
- 한번의 실행으로 구조를 얻고, 구조를 설계할 때 적절한 a-posteriori를 선택할 수 있도록 보조.
- 기존 "Evolving artificial neural networks"보다 **5개의 레이어를 추가했고, 네트워크를 제어하는 기능, 인코딩, 가중치 학습을 위한 low-level 최적화 프로세스를 통해 더 나은 성능을 제공**

### 구조

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled.png)


- NSGANetV1의 경우 Traing-off 전면에 걸친 구조를 설계
- low-level의 SGD(stochastic Gradient Descent)를 통해 최적화를 시키고 모델의 성능을 측정
- 검색은 앞서 말한 유전적연산(recombine 및 evolution algorithm)을 통해 분포를 추정

### Search Efficiency

- NAS는 모델 성능 평가를 위한 가중치 학습의 최적화에 목적
- 제한된 환경 안에서 활용성을 향상시키기 위해 NAS방식을 채택
- 이와 관련된 방식은 depth와 width를 줄여 프록시 모델을 만든다.

### Proxy model

- 최적화를 수행하기 위한 시간이 적게 소요
- 휴리스틱을 따라 모델을 구성하므로 예측과의 상관관계가 다소 낮다는 단점이 있음.
- 휴리스틱 이론 : 불충분한 시간이나 정보로 인하여 합리적인 판단을 할 수 없거나, 체계적이면서 합리적인 판단이 굳이 필요하지 않은 상황에서 빠르게 사용할 수 있게 구성된 간편추론의 방식

### General Framework

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled1.png)

### 인코딩

- 동일한 공간 해상도를 output으로 제공
- 공간 해상도가 절반으로 줄어든 정보를 반환하는 또 다른 방식은 Reduction
- 위와 같은 방식을 두 블록으로 구성하는데 이는 DAG(Directed Acyclic Graph)를 사용하여 구성

### 구조

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled2.png)

- 구조는 스택블록으로 구성
- 채널 수는 depth에 따라 증가
- 각 블록은 5개의 노드로 구성
- 각 노드는 동일한 블록 내에서 과거 블록 및 노드의 출력에 적용되는 양방향 계산

### 노드 구조

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled3.png)

- 첫번째 parent의 reduction 블록과 recombine하여 자손을 형성
- 한 parent노드가 무작위로 선택되어 다른 parent-level에 있는 다른 노드와 교환되는 방식
- 위 두가지 유형을 크로스오버하여 하위 구조를 효율적으로 교환

### Input & Operation Mutation

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled4.png)

- h1, h2는 ouput
- h3는 Polynomial mutation(PM)연산자를 통해 크로스오버하여 출력

### 하이퍼 파라메터 설정

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled5.png)

### 성능 측정

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled6.png)

- 이전 방법보다 효율적이고 정확함

### ETC

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled7.png)

- NSGANet을 ImageNet에서 이식이 가능
- 점선으로 표시된 부분은 multi-objective algorithm을 사용한 것

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled8.png)

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled9.png)

### CheXNet과의 비교

- 동일한 데이터셋(NIB) 사용

![Imgae Alt](assets/images/Multi-Objective Evolution/Untitled10.png)

### 결론

- NSGANet은 구조 구성 요소를 재조합하고 변형해 성능을 개선
- 베이시안 네트웤 모델을 통해 분포 추정을 통해 성공적으로 예측한 구조간의 패턴을 추출하여 성능을 향상
- EAs(진화 알고리즘)의 중요성을 강조
