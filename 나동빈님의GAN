# 나동빈님의 GAN: Generative Adversarial Networks (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)

- Link: https://www.youtube.com/watch?v=AVvlDmhHgC4

## 컴퓨터는 어떻게 존재하지 않는 그럴싸한 이미지를 만들어 낼까?

![캡처1](https://user-images.githubusercontent.com/59161837/104556917-b60de300-5683-11eb-8b7b-b558b3197049.PNG)



확률분포는 확률 변수가 특정한 값을 가질 확률을 나타내는 함수를 의미한다.

-> 확률분포는 이산확률분포와 연속확률분포가 있다.



## 이미지 데이터에 대한 확률분포

![캡처2](https://user-images.githubusercontent.com/59161837/104557022-e2296400-5683-11eb-8a65-3edf83e1813b.PNG)



-> 이미지 데이터는 다차원 특징 공간의 한 점으로 표현된다.

-> 이미지의 분포를 근사하는 모델을 학습할 수 있다.

-> 사람의 얼굴에는 통계적인 평균치가 존재할 수 있다.

-> 모델은 이를 수치적으로 표현할 수 있다.



## 생성 모델(Generative Models)은 실존하지 않지만 있을 법한 이미지를 생성할 수 있는 모델을 의미한다.

![캡처3](https://user-images.githubusercontent.com/59161837/104557159-1ac93d80-5684-11eb-93c2-05964a5e9d3c.PNG)



-> 이미지 데이터의 분포를 근사하는 모델 G를 만드는 것이 생성 모델의 목표이다.

-> 모델 G가 잘 동작한다는 의미는 원래 이미지들의 분포를 잘 모델링할 수 있다는 것을 의미한다.

-> 2014년에 제안된 Generaive Adversarial Networks (GAN)이 대표적이다.

-> 모델 G의 학습이 잘 되었다면 원본 데이터의 분포를 근사할 수 있다.

-> 학습이 잘 되었다면 통계적으로 평균적인 특징을 가지는 데이터를 쉽게 생성할 수 있다.



## Generative Adversarial Networks (GAN): 생성자(generator)와 판별자(discriminator) 두 개의 네트워크를 활용한 생성 모델이다.

-> 목적 함수(objective funtion)을 통해 생성자는 이미지 분포를 학습할 수 있다.

![캡처4](https://user-images.githubusercontent.com/59161837/104558244-dc348280-5685-11eb-8172-2693e0a5d609.PNG)



- V(D, G)라는 함수는 G값은 낮추려고 하고, D값은 높이려고 한다.
- x~pdata(x): pdata는 원본 데이터의 분포를 의미한다. x~pdata(x)는 원본 데이터에서 하나의 데이터인 x를 샘플로 꺼내겠다는 의미다.
- E[logD(x)]: 하나의 데이터인 x를 D()함수에 넣어서 나오는 기댓값 E[logD(x)]라는 의미다.
- z~pz(z): 생성자는 기본적으로 노이즈 벡터(z)로부터 입력을 받아서 새로운 이미지를 생성한다.  이때 pz는 하나의 노이즈를 샘플링할 수 있는 분포가 된다. z~pz(z)는 노이즈 벡터로부터 생성된 분포에서 랜덤하게 뽑인 하나의 노이즈z를 의미한다.
- E[log(1-D(G(z)))]: 하나의 노이즈인 z를 G()함수에 넣고 다시 D()함수에 넣은 값을 1에서 뺀 값에 log를 붙인 기댓값을 의미한다.

![캡처5](https://user-images.githubusercontent.com/59161837/104558466-31709400-5686-11eb-8ff1-784b13c999a6.PNG)



- Generator: 하나의 노이즈 벡터z를 받아서 새로운 data instance를 만든다.
- Discriminator: 이미지 x를 받아서 진짜 이미지는 1, 가짜 이미지는 0을 부여하는 방식으로 학습한다. 이 때 출력값은 확률 값이므로 0~1사이 값을 출력한다.

1. 노이즈 벡터 z가 들어와서 Generator를 거쳐 Fake image를 만들어 낸다.
2. Fake image는 Discriminator에 들어간다. 이 때 Generator는 Loss에서 Ez~pz(z)[log(1-D(G(z)))] 값을 낮추는 방향으로 학습한다.
3. 반면 Fake image와 Real image를 둘 다 받고 있는 Discriminator의 경우, Loss에서 Ex~pdata(x)[logD(x)] + Ez~pz(z)[log(1-D(G(z)))] 값을 높이는 방향으로 학습한다.



## GAN의 수렴 과정

-> GAN 공식의 목표는 생성자의 분포가 원본 학습 데이터의 분포를 잘 따를 수 있도록 반드는 것이다. (Pg -> Pdata), 즉 판별자는 생성자가 만든 이미지를 진짜인지 가짜인지 구분할 수 없기 때문에 D(G(z)) -> 1/2의 값을 내게 된다. 

![캡처6](https://user-images.githubusercontent.com/59161837/104560265-fb80df00-5688-11eb-8f59-4404d112caa8.PNG)



1. (a) -> (b) -> (c) -> (d) 는 시간에 흐름에 따라 표현한 것이다.
2. z에서 x로 가는 것은 노이즈 벡터z가 이미지 데이터 x로 맵핑된다는 것을 의미한다.
3. 처음에는(a) 생성 모델이 원본 데이터를 잘 학습하지 못하여서 원본 데이터의 분포보다 오른쪽으로 치우쳐져 있지만, 나중에는(d) 생성 모델이 원본 데이터의 분포와 일치하게 학습되어지는 것을 볼 수 있다.
4. 판별 모델의 분포 역시 (a)에서 (d)로 갈수록 1/2에 수렴하는 것을 볼 수 있다.



## 학습을 진행했을 때 도대체 왜 Pg(생성자의 분포)가 Pdata(원본 데이터의 분포)로 수렴할 수 있는 것일까? 이것이 GAN 논문에서 가장 핵심적으로 증명하는 부분이다.

![캡처7](https://user-images.githubusercontent.com/59161837/104562515-d04bbf00-568b-11eb-978c-0eeabbd2fde4.PNG)



실제로 학습을 진행했을 때 Pg가 Pdata로 수렴할까? -> Global Optimality를 증명하여 매 상황에 대해 생성자와 판별자가 어떤 값으로 Global Optimality를 가지는지 보자.

명제 1: G가 고정된 상황에서 V(G, D)의 Optimum Point는 DG*(x) = Pdata(x) / (Pdata(x) + Pg(x)) 라고 할 수 있다.

![캡처8](https://user-images.githubusercontent.com/59161837/104563458-03428280-568d-11eb-94f1-88c43de33210.PNG)



명제 2: 생성자의 Global Optimum Point 는 Pg = Pdata이다.

C(G) = maxDV(G,D) = -log(4) 값(Global Optimum Point)을 얻을 수 있는 유일한 포인트는 Pg = Pdata 인 경우임을 알 수 있다. 따라서 생성자(G)는 판별자(D)가 매번 잘 수렴해서 Global Optimum Point를 잘 가지고 있다고 가정한 상태에서 생성자가 잘 학습된다면 -log(4)로 수렴하고 이것이 곧 Pg = Pdata라는 의미이다.

물론 이 증명은 생성자와 판별자, 각각에 대해서 Global Optimum Point가 존재할 수 있다는 것을 증명한 내용이고, 생성자와 판별자가 Global Optimum Point에 실제로 도달할 수 있는가? 에 대한 내용은 엄밀히 말하면 다른 내용이다.

-> 원본 GAN 논문에서는 자세히 언급하고 있지 않지만, 사실 GAN은 학습이 어려운 네트워크 중 하나이다. 따라서 원본 GAN 논문이 나온 이후에 다른 논문들을 통해서 학습에 안정성을 더할 수 있는 다양한 테크닉들이 나오기도 했다.



## GAN 알고리즘

![캡처9](https://user-images.githubusercontent.com/59161837/104565228-51588580-568f-11eb-9363-e09fac412988.PNG)

1. 학습 반복 횟수만큼 반복하도록 만든다 -> epoch이라고 표현함
2. k번의 Discriminator 학습 후 Generator을 학습한다.
3. Discriminator을 학습할 때, m개의 노이즈와 m개의 원본 데이터를 뽑는다. 다음으로 Loss값을 구해 높이는 방향으로 학습시킨다.
4. Generator을 학습하기 위해, m개의 노이즈를 뽑아  Loss값을 구해 낮추는 방향으로 학습시킨다.



- Not cherry-picked: 논문에 넣어둔 생성 이미지는 별도로 선별해서 넣은게 아니고 랜덤으로 뽑아서 넣은 것이다.
- Not memorized the training set: 학습 데이터를 단순히 암기한 것이 아니라 새로운 생성 이미지로 만든 것이다.
- Competitive with the better generative models: 본 논문이 나오기 이전 시점까지의 다른 생성 모델들과 비교했을 때 충분히 좋은 성능이 나온다는 것을 보여주었다.
- Images represent sharp: 오토인코더 계열의 다른 생성 네트워크와 비교했을 때 상대적으로 더 sharp한 이미지를 생성한다.
