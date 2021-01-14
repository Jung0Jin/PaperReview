# Generative Adversarial Nets


- Link: https://arxiv.org/pdf/1406.2661.pdf



## 1. Abstract

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. 

우리는 적대적인 프로세스를 통해 생성 모델을 추정하기 위한 새로운 프레임워크를 제안한다. 여기서 우리는 두 가지 모델, 즉 데이터 분포를 캡처하는 생성 모델 G와 샘플이 G가 아닌 훈련 데이터에서 나왔을 확률을 추정하는 판별 모델 D를 동시에 훈련한다.



The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. 

G에 대한 훈련 절차는 D가 실수할 확률을 최대화하는 것이다. 이 프레임워크는 최소 2인용 게임에 해당한다. 임의의 함수 G와 D의 공간에는 G가 훈련 데이터 분포를 복구하고 D가 1/2인 고유한 솔루션이 존재한다.



In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples. 

G와 D가 다층 퍼셉트론에 의해 정의되는 경우, 전체 시스템은 backpropagation으로 훈련될 수 있다. 훈련 또는 샘플 생성 중에 Markov chains 또는 unrolled approximate inference networks가 필요하지 않다. 실험은 생성된 샘플의 정성적 및 정량적 평가를 통해 프레임워크의 잠재력을 입증한다.



## 2. Conclusions and future work

This framework admits many straightforward extensions: 



이 프레임워크는 많은 간단한 확장을 허용한다.

1. A conditional generative model p(x|c) can be obtained by adding C as input to both G and D 

   조건 생성 모델 p(x|c)는 C를 G와 D에 모두 입력으로 추가함으로써 얻을 수 있다. 

   

2. Learned approximate inference can be performed by training an auxiliary network to predict Z given X. This is similar to the inference net trained by the wake-sleep algorithm [15] but with the advantage that the inference net may be trained for a fixed generator net after the generator net has finished training.

   학습된 approximate inference는 X가 주어진 Z를 예측하기 위해 보조 네트워크를 훈련시킴으로써 수행할 수 있다. 이는 wake-sleep algorithm에 의해 훈련되는 inference net과 유사하지만, generator net가 훈련을 마친 후 fixed generator net에 대해 inference net가 훈련될 수 있다는 장점이 있다.

   

3. One can approximately model all conditionals p(xs | xnots) where S is a subset of the indices of x by training a family of conditional models that share parameters. Essentially, one can use adversarial nets to implement a stochastic extension of the deterministic MP-DBM [11]. 

   모든 조건 p(xs|xnots)를 근사적으로 모델링할 수 있다. 여기서 S는 매개변수를 공유하는 조건 모델군을 훈련함으로써 x의 지수의 하위 집합이다. 본질적으로, deterministic MP-DBM의 확률적 확장을 구현하기 위해 adversarial nets을 사용할 수 있다.

   

4. Semi-supervised learning: features from the discriminator or inference net could improve performance of classifiers when limited labeled data is available. 

   Semi-supervised learning: discriminator 또는 inference net의 features는 제한된 labeled data를 이용할 수 있을 때, classifiers의 성능을 향상시킬 수 있다.

   

5. Efficiency improvements: training could be accelerated greatly by divising better methods for coordinating G and D or determining better distributions to sample Z from during training. 

   효율성 개선: G와 D를 조정하는 더 나은 방법 또는 표본 Z에 대한 더 나은 분포를 training중에 결정함으로써, training 시간을 크게 단축할 수 있다.

   

This paper has demonstrated the viability of the adversarial modeling framework, suggesting that these research directions could prove useful.

본 논문은 adversarial modeling framework의 실행 가능성을 입증했으며, 이러한 연구 방향이 유용할 수 있음을 시사한다.



## 3. Introduction

The promise of deep learning is to discover rich, hierarchical models [2] that represent probability distributions over the kinds of data encountered in artificial intelligence applications, such as natural images, audio waveforms containing speech, and symbols in natural language corpora. 

deep learning의 promise는 자연 이미지, 음성을 포함하는 오디오 파형, 자연어 말뭉치의 기호와 같은 인공 지능 애플리케이션에서 마주치는 데이터 종류에 대한 확률 분포를 나타내는 rich, hierarchical models을 발견하는 것이다.



So far, the most striking successes in deep learning have involved discriminative models, usually those that map a high-dimensional, rich sensory input to a class label [14, 22]. These striking successes have primarily been based on the backpropagation and dropout algorithms, using piecewise linear units [19, 9, 10] which have a particularly well-behaved gradient . 

지금까지 deep learning에서 가장 두드러진 성공은 차별적인 모델(discriminative models), 보통 고차원적이고 풍부한 감각 입력(high-dimensional, rich sensory input)을  클래스 라벨에 매핑하는, 모델을 포함한다. 이러한 놀라운 성공은 주로 backpropagation, dropout algorithms에 기초해 왔으며, 특히 잘 동작하는 경사도를 가진 단편적인 선형 단위(piecewise linear units)를 사용하였다.



Deep generative models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context. We propose a new generative model estimation procedure that sidesteps these difficulties.

Deep generative models은 최대우도추정(maximum likelihood estimation) 및 관련 전략에서 발생하는 많은 난해한 확률론적 계산(intractable probabilistic computations)을 근사화하는 어려움과 생성적 맥락에서 단편적인 선형 단위의 이점(the benefits of piecewise linear untis in the generative context)을 활용하기 어렵기 때문에 영향을 덜 받았다. 우리는 이러한 어려움을 피하는 새로운 생성 모델 추정 절차(new generative model estimation procedure)를 제안한다.



In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution. 

제안된 adversarial nets framework에서, 생성 모델(generative model)은 적과 비교된다(pitted against an adversary): 표본이 모형 분포에서 추출되었는지 또는 데이터 분포에서 추출되었는지 여부를 확인하는 방법을 학습하는 판별 모형



The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. 

생성 모델(generative model)은 위조 화폐를 생산하여 탐지 없이 사용하려고 하는 위조 화폐팀과 유사하다고 생각할 수 있고, 판별 모델(discriminative model)은 위조 화폐를 탐지하려고 하는 경찰과 유사하다.



Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles.

이 게임에서의 경쟁은 두 팀에게 모조품이 진짜 물품에서 구분이 안 될 때까지 그들의 방법을 개선하도록 강요한다.



This framework can yield specific training algorithms for many kinds of model and optimization algorithm. In this article, we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron. 

이 프레임워크는 많은 종류의 모델 및 최적화 알고리즘에 대한 특정 훈련 알고리즘을 산출할 수 있다. 이  논문에서, 우리는 생성 모델이 multilayer perceptron을 통해 무작위 노이즈를 전달하여 샘플을 생성하는 특별한 경우를 탐구하며, 판별 모델도 multilayer perceptron이다.



We refer to this special case as adversarial nets. In this case, we can train both models using only the highly successful backpropagation and dropout algorithms [17] and sample from the generative model using only forward propagation. No approximate inference or Markov chains are necessary

우리는 이 특별한 경우를 adversarial nets라고 부른다. 이 경우, 우리는 매우 성공적인 backpropagation 및 dropout algorithms만 사용하여 두 모델을 모두 훈련할 수 있고 forward propagation만을 사용하여 생성 모델의 샘플도 훈련할 수 있다. approximate inference나 Markov chains은 필요하지 않다.



## 4. Related work

An alternative to directed graphical models with latent variables are undirected graphical models with latent variables, such as restricted Boltzmann machines (RBMs) [27, 16], deep Boltzmann machines (DBMs) [26] and their numerous variants. 

잠재 변수(latent variables)를 가진 지시된 그래픽 모델의 대안(An alternative to directed graphical models)은 제한된 볼츠만 기계(restricted Boltzmann machines, RBMs), 심층 볼츠만 기계(deep Boltzmann machines, DBMs) 및 수많은 변형과 같은 잠재 변수를 가진 무방향 그래픽 모델이다.



The interactions within such models are represented as the product of unnormalized potential functions, normalized by a global summation/integration over all states of the random variables. 

그러한 모델 내의 상호작용은 모든 랜덤 변수의 상태에 대한 전역 합계/통합에 의해 정규화된 비정규화된 잠재적 함수의 산물로 표현된다.



This quantity (the partition function) and its gradient are intractable for all but the most trivial instances, although they can be estimated by Markov chain Monte Carlo (MCMC) methods. Mixing poses a significant problem for learning algorithms that rely on MCMC [3, 5].

Markov chain Monte Carlo(MCMC) 방법으로 추정할 수 있지만, 이 수량(파티션 함수)과 그 gradient는 가장 사소한 경우를 제외하고 모두 다루기 어렵다. 혼합은 MCMC에 의존하는 알고리즘 학습에 중요한 문제를 제기한다. 
