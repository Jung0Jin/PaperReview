# Generative Adversarial Nets

- Link: https://arxiv.org/pdf/1406.2661.pdf

## 1. Abstract

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. 

우리는 적대적인 프로세스를 통해 생성 모델을 추정하기 위한 새로운 프레임워크를 제안한다. 여기서 우리는 두 가지 모델, 즉 데이터 분포를 캡처하는 생성 모델 G와 샘플이 G가 아닌 훈련 데이터에서 나왔을 확률을 추정하는 판별 모델 D를 동시에 훈련한다.

The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. 

G에 대한 훈련 절차는 D가 실수할 확률을 최대화하는 것이다. 이 프레임워크는 최소 2인용 게임에 해당한다. 임의의 함수 G와 D의 공간에는 G가 훈련 데이터 분포를 복구하고 D가 1/2인 고유한 솔루션이 존재한다.

In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples. 

G와 D가 다층 퍼셉트론에 의해 정의되는 경우, 전체 시스템은 backpropagation으로 훈련될 수 있다. 훈련 또는 샘플 생성 중에 Markov chains 또는 unrolled approximate inference networks가 필요하지 않다. 실험은 생성된 샘플의 정성적 및 정량적 평가를 통해 프레임워크의 잠재력을 입증한다.

## 2. Conclusion

This framework admits many straightforward extensions: 

이 프레임워크는 많은 간단한 확장을 허용한다.

1. A conditional generative model p(x|c) can be obtained by adding c as input to both G and D 

   조건 생성 모델 p(x|c)는 c를 G와 D에 모두 입력으로 추가함으로써 얻을 수 있다.

2. Learned approximate inference can be performed by training an auxiliary network to predict z given x. This is similar to the inference net trained by the wake-sleep algorithm [15] but with the advantage that the inference net may be trained for a fixed generator net after the generator net has finished training. 

   학습된 approximate inference는 x가 주어진 z를 예측하기 위해 보조 네트워크를 훈련함으로써 수행될 수 있다. 이는 wake-sleep algorithm에 의해 훈련되는 inference net과 유사하지만, generator net가 훈련을 마친 후 fixed generator net에 대해 inference net가 훈련될 수 있다는 장점이 있다.
