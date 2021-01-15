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



Deep belief networks (DBNs) [16] are hybrid modelss containing a single undirected layer and several directed layers. While a fast approximate layer-wise training criterion exists, DBNs incur the computational difficulties associated with both undirected and directed models.

Deep belief networks (DBNs)는 단일 무방향 레이어(a single undirected layer)와 여러 방향 레이어(several directed layers)를 포함하는 하이브리드 모델이다. 빠른 근사 계층별 훈련 기준(approximate layer-wise training criterion)이 존재하지만, DBN은 무방향 및 방향 모델과 관련된 계산상의 어려움을 초래한다.



Alternative criteria that do not approximate or bound the log-likelihood have also been proposed, such as score matching [18] and noise-contrastive estimation (NCE) [13]. Both of these require the learned probability density to be analytically specified up to a normalization constant. 

점수 일치(score matching) 및 노이즈 대비 추정(noise-contrastive estimation, NCE) 같이 로그 우도(log-likelihood)를 approximate 또는 bound하지 않는 대체 기준(Alternative criteria)도 제안되었다. 이 두 가지 모두 정규화 상수까지 분석적으로 지정된 학습된 확률 밀도(learned probability density)가 필요하다.



Note that in many interesting generative models with several layers of latent variables (such as DBNs and DBMs), it is not even possible to derive a tractable unnormalized probability density. 

latent variables (such as DBNs and DBMs)의 여러 레이어가 있는 흥미로운 생성 모델에서는 다루기 쉬운 비정규화 확률 밀도(unnormalized probability density)를 도출할 수 없다.



Some models such as denoising auto-encoders [30] and contractive autoencoders have learning rules very similar to score matching applied to RBMs. In NCE, as in this work, a discriminative training criterion is employed to fit a generative model. 

denoising auto-encoders 및 contractive autoencoders 같은 일부 모델은 RBMs에 적용되는 score matching 과 매우 유사한 학습 규칙을 가지고 있다. NCE에서, 본 연구에서와 같이, discriminative training criterion은 생성 모델에 적합하도록 사용된다.



However, rather than fitting a separate discriminative model, the generative model itself is used to discriminate generated data from samples a fixed noise distribution. 

그러나 생성 모델 자체는 별도의 판별 모델을 장착하는 대신, 생성된 데이터를 표본에서 고정된 noise 분포를 구별하는 데 사용된다.



Because NCE uses a fixed noise distribution, learning slows dramatically after the model has learned even an approximately correct distribution over a small subset of the observed variables.

NCE는 고정된 noise 분포를 사용하기 때문에, 모델이 관찰된 변수의 작은 부분 집합에 대해 거의 정확한 분포도 학습한 후 학습 속도가 크게 느려진다.



Finally, some techniques do not involve defining a probability distribution explicitly, but rather train a generative machine to draw samples from the desired distribution. This approach has the advantage that such machines can be designed to be trained by back-propagation. 

마지막으로, 일부 기법은 확률 분포를 명시적으로 정의하지 않고 원하는 분포에서 표본을 추출하도록 생성 기계를 훈련시킨다. 이 접근 방식은 그러한 기계가 backpropagation에 의해 훈련되도록 설계될 수 있다는 이점이 있다.



Prominent recent work in this area includes the generative stochastic network (GSN) framework [5], which extends generalized denoising auto-encoders [4]: both can be seen as defining a parameterized Markov chain, i.e., one learns the parameters of a machine that performs one step of a generative Markov chain. 

이 분야의 두드러진 최근 연구에는 일반화된 denoising auto-encoders를 확장하는 생성 확률 네트워크(the generative stochastic network, GSN) 프레임워크가 포함된다. 둘 다 매개 변수화된 Markov chain을 정의하는 것으로 볼 수 있다. 즉, a generative Markov chain의 한 단계를 수행하는 기계의 매개 변수를 학습하는 것으로 볼 수 있다.



Compared to GSNs, the adversarial nets framework does not require a Markov chain for sampling. Because adversarial nets do not require feedback loops during generation, they are better able to leverage piecewise linear units [19, 9, 10], which improve the performance of backpropagation but have problems with unbounded activation when used ina feedback loop. 

GSNs와 비교하여, the adversarial nets framework는 샘플링을 위해 Markov chain을 요구하지 않는다. adversarial nets은 생성 중에 피드백 루프를 필요로 하지 않기 때문에, backpropagation의 성능을 향상시키지만 피드백 루프에 사용될 때 무한 활성화 문제(probelms with unbounded activation)가 있는 단편적인 선형 단위(piecewise linear units)를 더 잘 활용할 수 있다.



More recent examples of training a generative machine by back-propagating into it include recent work on auto-encoding variational Bayes [20] and stochastic backpropagation [24].

backpropagation를 통해 generative machine을 훈련시키는 더 최근의 예에는 auto-encoding variational Bayes, stochastic backpropagation에 대한 최근의 연구가 포함된다.



## 5. Adversarial nets

The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons. To learn the generator’s distribution Pg over data X, we define a prior on input noise variables Pz(z), then represent a mapping to data space as G(z; θg), where G is a differentiable function represented by a multilayer perceptron with parameters θg.

the adversarial modeling framework는 모델이 둘 다  multilayer perceptrons일 때 적용하기가 가장 간단하다. 데이터 X에 대한 generator's distribution Pg를 학습하기 위해 입력 noise variables Pz(z)를 정의한 다음 데이터 공간에 대한 매핑을 G(z; θg)로 나타내며, 여기서 G는 매개 변수 θg를 가진 multilayer perceptron으로 표현되는 차별화 가능한 함수(differentiable function)이다.



 We also define a second multilayer perceptron D(x; θd) that outputs a single scalar. D(x) represents the probability that X came from the data rather than Pg. We train D to maximize the probability of assigning the correct label to both training examples and samples from G. We simultaneously train G to minimize log(1 − D(G(z))):

또한 a single scalar를 출력하는 두 번째 multilayer perceptron D(x; θd)을 정의한다. D(x)는 X가 Pg가 아니라 데이터에서 나왔을 확률을 나타낸다. 우리는 D를 훈련 예제와 G의 표본 모두에 올바른 레이블을 지정할 확률을 최대화하기 위해 훈련한다. G를 동시에 훈련하여 log(1-D(G(z)))를 최소화한다.



In other words, D and G play the following two-player minimax game with value function V (G, D):

즉, D와 G는 value function V(G, D)로 다음과 같은 2인용 minimax game을 한다.



![캡처](https://user-images.githubusercontent.com/59161837/104677146-40aa1d00-572c-11eb-858d-d014363a8fd5.PNG)



In the next section, we present a theoretical analysis of adversarial nets, essentially showing that the training criterion allows one to recover the data generating distribution as G and D are given enough capacity, i.e., in the non-parametric limit. 

다음 섹션에서는 adversarial nets에 대한 이론적 분석을 제시하며, 기본적으로 training criterion이 G와 D가 비모수 한계에서 충분한 capacity를 제공하므로 데이터 생성 분포를 복구할 수 있음을 보여준다.



See Figure 1 for a less formal, more pedagogical explanation of the approach. In practice, we must implement the game using an iterative, numerical approach. Optimizing D to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting. Instead, we alternate between k steps of optimizing D and one step of optimizing G. 

접근법에 대한 덜 공식적이고 교육적인 설명은 Figure 1을 참조하라. 실제로, 우리는 반복적이고 수치적인 접근 방식을 사용하여 게임을 구현해야 한다. 훈련의 내부 루프에서 D를 완료하도록 최적화하는 것은 계산적으로 금지되며, 유한 데이터셋에서는 과적합이 발생할 수 있다. 대신, 우리는 D를 최적화하는 k 단계와 G를 최적화하는 한 단계를 번갈아 한다.



This results in D being maintained near its optimal solution, so long as G changes slowly enough. This strategy is analogous to the way that SML/PCD [31, 29] training maintains samples from a Markov chain from one learning step to the next in order to avoid burning in a Markov chain as part of the inner loop of learning. The procedure is formally presented in Algorithm 1.

따라서 G가 충분히 느리게 변하는 한 D는 최적 솔루션 가까이 유지된다. 이 전략은 학습의 내부 루프의 일부로 Markov chain에 연소되는 것을 피하기 위해 SML/PCD training이 한 학습 단계에서 다음 학습 단계로 Markov chain의 샘플을 유지하는 방법과 유사하다. 이 절차는 Algorithm 1에 공식적으로 제시되어 있다.



In practice, equation 1 may not provide sufficient gradient for G to learn well. Early in learning, when G is poor, D can reject samples with high confidence because they are clearly different from the training data. In this case, log(1 − D(G(z))) saturates. Rather than training G to minimize log(1 − D(G(z))) we can train G to maximize log D(G(z)). This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning.

실제로 equation 1은 G가 잘 학습하기에 충분한 gradient를 제공하지 않을 수 있다. 학습 초기에 G가 불량할 때 D는 훈련 데이터와 분명히 다르기 때문에 신뢰도가 높은 표본을 기각할 수 있다. 이 경우 log(1-D(G(z)))가 포화 상태가 된다. log(1-D(G(z)))를 최소화하기 위해 G를 훈련하는 대신 G를 훈련시켜 logD(G(z))를 최대화할 수 있다. 이 목적 함수는 G와 D의 the same fixed point of the dynamics를 얻지만 학습 초기에 훨씬 더 강한 gradients를 제공한다.

![캡처2](https://user-images.githubusercontent.com/59161837/104677917-a8ad3300-572d-11eb-8d4f-3543fab52f46.PNG)

Figure 1: Generative adversarial nets은 데이터 생성 분포(the data generating distribution)(black, dotted line) px에서 생성된 분포 (G) (green, solid line)의 표본을 구별하도록 discriminative distribution(D, blue, dashed line)를 동시에 업데이트함으로써 훈련된다. 아래쪽 수평선은 z가 샘플링되는 도메인이며, 이 경우 균일하게 표시된다. 위의 수평선은 x 영역의 일부다. 위쪽 화살표는 매핑 x = G(z)가 변환된 표본에 대해 균일하지 않은 분포 pg를 부과하는 방법을 보여준다. G는 고밀도의 지역에서 수축하고 pg밀도가 낮은 지역에서 확장된다. (a) 수렴에 가까운  adversarial pair로 간주한다. pg는 pdata와 유사하며 D는 부분적으로 정확한 classifier다. (b) algorithm D의 내부 루프에서는 샘플과 데이터를 구별하도록 훈련되어 D*(x)=pdata(x)/(pdata(x)+pg(x))로 수렴한다. (c) G로 업데이트한 후, D의 gradient는 G(z)가 데이터로 분류될 가능성이 높은 지역으로 흐르도록 유도했다. (d) 몇 단계의 교육 후, G와 D는 capacity사 충분한 경우, pg와 = pdata로 인해 둘 다 개선될 수 없는 시점에 도달하게 된다. discriminator은 두 분포를 구분할 수 없다. 즉, D(x) = 1/2.



## 6. Theoretical Results

 The generator G implicitly defines a probability distribution pg as the distribution of the samples G(z) obtained when z ∼ pz. Therefore, we would like Algorithm 1 to converge to a good estimator of pdata, if given enough capacity and training time. The results of this section are done in a nonparametric setting, e.g. we represent a model with infinite capacity by studying convergence in the space of probability density functions. We will show in section 4.1 that this minimax game has a global optimum for pg = pdata. We will then show in section 4.2 that Algorithm 1 optimizes Eq 1, thus obtaining the desired result.

생성자 G는 z~pz에서 얻은 표본 G(z)의 분포로 확률 분포 Pg를 암시적으로 정의한다. 따라서 충분한 capacity와 training time이 주어지면 Algorithm 1이 적절한 pdata의 estimator로 수렴되기를 바란다. 이 섹션의 결과는 비모수 설정에서 수행된다. 예를 들어, 우리는 확률밀도함수의 공간에서 수렴을 연구하여 무한한 용량을 가진 모델을 가진 모델을 나타낸다. 4.1절에서 이 minimax game이 pg=pdata에 대한 global optimum을 가지고 있다는 것을 보여주겠다. 그런 다음 4.2절에서 Algorithm 1이 Eq 1을 최적화하여 원하는 결과를 얻는다는 것을 보여줄 것이다.

![캡처3](https://user-images.githubusercontent.com/59161837/104682339-c54e6880-5737-11eb-9672-ada08d2bdebc.PNG)



![캡처4](https://user-images.githubusercontent.com/59161837/104682379-dbf4bf80-5737-11eb-8a75-c5dc1cc91916.PNG)



![캡처5](https://user-images.githubusercontent.com/59161837/104682411-f5960700-5737-11eb-960a-9a7d54804845.PNG)



![캡처6](https://user-images.githubusercontent.com/59161837/104682466-13636c00-5738-11eb-979a-2b55244e12a8.PNG)

In practice, adversarial nets represent a limited family of pg distributions via the function G(z; θg), and we optimize θg rather than pg itself. Using a multilayer perceptron to define G introduces multiple critical points in parameter space. However, the excellent performance of multilayer perceptrons in practice suggests that they are a reasonable model to use despite their lack of theoretical guarantees.

실제로, adversarial nets는 함수 G(z; θg)를 통한 제한된 pg 분포군을 나타내며, 우리는 pg 자체보다는 θg를 최적화한다. multilayer perceptron을 사용하여 G를 정의하면 매개 변수 공간에 여러 임계점이 도입된다. 그러나, 실제로 multilayer perceptron의 우수한 성능은 이론적인 보증이 없음에도 불구하고 그것들이 사용하기에 합리적인 모델임을 시사한다.



## 7. Experiments

We trained adversarial nets an a range of datasets including MNIST[23], the Toronto Face Database (TFD) [28], and CIFAR-10 [21]. The generator nets used a mixture of rectifier linear activations [19, 9] and sigmoid activations, while the discriminator net used maxout [10] activations. Dropout [17] was applied in training the discriminator net. While our theoretical framework permits the use of dropout and other noise at intermediate layers of the generator, we used noise as the input to only the bottommost layer of the generator network.

우리는 MNIST, Toronto Face Database(TFD) 및 CIFAR-10을 포함한 다양한 데이터 세트를 adversarial nets로 훈련시켰다. generator nets는 rectifier linear activations과 sigmoid activations의 혼합물을 사용하는 반면, discriminator net은 maxout activations을 사용했다. Dropout은 discriminator net의 training에 적용되었다. 우리의 이론적 프레임워크는 generator의 중간 계층(intermediate layers)에서 Drouout과 다른 noise를 사용하는 것을 허용하지만, 우리는 generator network의 최하위 계층(bottommost layer)에만 대한 입력으로 noise를 사용했다.



We estimate probability of the test set data under pg by fitting a Gaussian Parzen window to the samples generated with G and reporting the log-likelihood under this distribution. The σ parameter of the Gaussians was obtained by cross validation on the validation set. This procedure was introduced in Breuleux et al. [8] and used for various generative models for which the exact likelihood is not tractable [25, 3, 5]. Results are reported in Table 1. This method of estimating the likelihood has somewhat high variance and does not perform well in high dimensional spaces but it is the best method available to our knowledge. Advances in generative models that can sample but not estimate likelihood directly motivate further research into how to evaluate such models.

G로 생성된 샘플에 Gaussian Parzen window를 적합시키고 이 분포에서 log-likelihood를 보고함으로써 pg 아래 테스트 세트 데이터의 확률을 추정한다. Gaussians의 σ 매개변수는 검증 세트에서 교차 검증을 통해 얻었다. 이 절차는 Breuleux 등에 도입되었다. 정확한 likelihood가 추적 불가능한 다양항 생성 모델에 사용된다. 결과는 Table 1.에 보고된다. 이 likelihood 추정 방법은 분산이 다소 높으며 고차원 공간에서는 잘 수행되지 않지만 우리가 알 수 있는 최선의 방법이다. 표본은 추출할 수 있지만 likelihood를 직접 추정할 수 없는 생성 모델의 advances는 그러한 모델을 평가하는 방법에 대한 추가 연구에 동기를 부여한다.

![캡처7](https://user-images.githubusercontent.com/59161837/104686873-71488180-5741-11eb-8611-d78e7eb1c7af.PNG)

Table 1: Parzen window-based log-likelihood estimates. MNIST에 보고된 숫자는 표본에 대해 계산된 평균의 표준 오차와 함께 검정 세트에서 표본의 평균 log-likelihood이다. TFD에서는 데이터 집합의 폴드에 걸쳐 표준 오차를 계산했으며, 각 폴드의 유효성 검사 세트를 사용하여 다른 σ를 선택했다. TFD에서 각 폴드에 대해 σ를 교차 검증하고 각 폴드에 대한 평균 log-likelihood를 계산했다. MNIST의 경우, 우리는 (이진형이 아닌) 실제 가치 데이터 세트의 다른 모델과 비교한다.



In Figures 2 and 3 we show samples drawn from the generator net after training. While we make no claim that these samples are better than samples generated by existing methods, we believe that these samples are at least competitive with the better generative models in the literature and highlight the potential of the adversarial framework.

Figure 2, 3에는 training 후 generator net에서 추출한 샘플이 나와 있다. 우리는 이러한 샘플이 기존 방법에 의해 생성된 샘플보다 낫다고 주장하지는 않지만, 우리는 이러한 샘플이 문헌에서 더 나은 생성 모델과 최소한 경쟁적이며 adversarial framework의 잠재력을 강조한다.

![캡처8](https://user-images.githubusercontent.com/59161837/104705316-191f7880-575d-11eb-989a-8f929e3637a8.PNG)

Figure 2: 모델의 샘플 시각화 맨 오른쪽 열에는 모형이 training set를 기억하지 못한다는 것을 보여주기 위해 인접 표본의 가장 가까운 training 예가 표시된다. 샘플은  cherry-picked한 것이 아니라 공정한 random draws다. deep generative models의 대부분의 다른 시각화와는 달리 이러한 이미지는 숨겨진 단위의 표본이 주어진 조건부 평균이 아니라 모델 분포의 실제 표본을 보여준다. 더욱이, 이러한 샘플들은 샘플 추출 과정이 Markov chain mixing에 의존하지 않기 때문에 상관 관계가 없다.

![캡처9](https://user-images.githubusercontent.com/59161837/104714539-a9af8600-5768-11eb-9bdc-b8e2f6c51eb5.PNG)

## 8. Advantages and disadvantages

This new framework comes with advantages and disadvantages relative to previous modeling frameworks. The disadvantages are primarily that there is no explicit representation of pg(x), and that D must be synchronized well with G during training (in particular, G must not be trained too much without updating D, in order to avoid “the Helvetica scenario” in which G collapses too many values of z to the same value of x to have enough diversity to model pdata), much as the negative chains of a Boltzmann machine must be kept up to date between learning steps. The advantages are that Markov chains are never needed, only backprop is used to obtain gradients, no inference is needed during learning, and a wide variety of functions can be incorporated into the model. Table 2 summarizes the comparison of generative adversarial nets with other generative modeling approaches. The aforementioned advantages are primarily computational. Adversarial models may also gain some statistical advantage from the generator network not being updated directly with data examples, but only with gradients flowing through the discriminator. This means that components of the input are not copied directly into the generator’s parameters. Another advantage of adversarial networks is that they can represent very sharp, even degenerate distributions, while methods based on Markov chains require that the distribution be somewhat blurry in order for the chains to be able to mix between modes.

이 새로운 프레임워크에는 이전 모델링 프레임워크와 관련된 장점과 단점이 함께 제공된다. 단점은 주로 pg(x)의 명시적 표현이 없고, 훈련 중에 D가 G와 잘 동기화되어야 한다는 것이다(특히, G는 D를 업데이트하지 않고 너무 많이 훈련되어서는 안 된다. G가 너무 많은 z값을 x의 동일한 값으로 축소하여 pdata를 모델링하기에 충분한 다양성을 갖는 "the Helvetica scenario"를 피하기 위해서이다.) 볼츠만 기계의 마이너스 체인(negative chains of Boltzmann machine)은 학습 단계 사이에 최신 상태로 유지되어야 한다. 장점은 Markov chains은 절대 필요하지 않으며, 오직 backpropagation만 gradients를 얻기 위해 사용되며, 학습 중에 추론이 필요하지 않으며, 다양한 기능을 모델에 통합할 수 있다는 것이다. Table 2는 generative adversarial nets와 다른 생성적 모델링 접근법의 비교를 요약한다. 앞서 언급한 장점들은 주로 계산적이다. 또한 adversarial 모델은 데이터 예제로 직접 업데이트되지 않고 discriminator를 통과하는 gradients에서만 generator network에서 통계적 이점을 얻을 수 있다. 이는 입력의 구성 요소가 generator의 매개 변수에 직접 복사되지 않음을 의미한다. adversarial networks의 또 다른 장점은 Markov chain에 기초한 방법은 chains이 모드들 사이에 혼합될 수 있도록 분포가 다소 흐릿할 것(blurry)을 요구하는 반면, 매우 날카롭고(very sharp) 심지어 퇴화된 분포(even degenerate distributions)를 나타낼 수 있다는 것이다.
