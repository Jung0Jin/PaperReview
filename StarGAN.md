# StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

- Link: https://arxiv.org/pdf/1711.09020.pdf

![캡처](https://user-images.githubusercontent.com/59161837/105477426-2dc3b980-5ce5-11eb-992e-86f84449ec3b.PNG)



## 1. Abstract

Recent studies have shown remarkable success in image-to-image translation for two domains. However, existing approaches have limited scalability and robustness in handling more than two domains, since different models should be built independently for every pair of image domains. 

최근 연구는 두 도메인에 대한 image-to-image에서 주목할 만한 성공을 보여주었다. 그러나 기존 접근 방식은 두 개 이상의 도메인을 처리할 때 확장성(scalability)과 견고성(robustness)이 제한되는데, 이는 서로 다른 모델이 모든 이미지 도메인 쌍에 대해 독립적으로 구축되어야 하기 때문이다.



To address this limitation, we propose StarGAN, a novel and scalable approach that can perform image-to-image translations for multiple domains using only a single model. Such a unified model architecture of StarGAN allows simultaneous training of multiple datasets with different domains within a single network. 

이 한계를 해결하기 위해, 우리는 단일 모델만을 사용하여 여러 도메인에 대해 image-to-image translation을 수행할 수 있는 새롭고 확장 가능한 접근 방식인 StarGAN을 제안한다. 이러한 StarGAN의 통합 모델 아키텍처는 단일 네트워크 내에서 서로 다른 도메인을 가진 여러 데이터 세트의 동시 훈련을 가능하게 한다.



This leads to StarGAN’s superior quality of translated images compared to existing models as well as the novel capability of flexibly translating an input image to any desired target domain. We empirically demonstrate the effectiveness of our approach on a facial attribute transfer and a facial expression synthesis tasks.

이는 기존 모델에 비해 StarGAN의 우수한 변환 이미지 품질과 입력 이미지를 원하는 모든 대상 도메인으로 유연하게 변환하는 새로운 기능으로 이어진다. 우리는 얼굴 전달 및 얼굴 표정 작업에 대한 접근 방식의 효과를 경험적으로 보여준다.



## 2. Conclusion

In this paper, we proposed StarGAN, a scalable image-to-image translation model among multiple domains using a single generator and a discriminator. Besides the advantages in scalability, StarGAN generated images of higher visual quality compared to existing methods [16, 23, 33], owing to the generalization capability behind the multi-task learning setting. In addition, the use of the proposed simple mask vector enables StarGAN to utilize multiple datasets with different sets of domain labels, thus handling all available labels from them. We hope our work to enable users to develop interesting image translation applications across multiple domains.

본 논문에서는 단일 생성기와 판별기를 사용하여 여러 도메인 사이에서 확장 가능한 image-to-image translation 모델인 StarGAN을 제안하였다. 확장성의 장점 외에도, StarGAN은 다중 작업 학습 설정(multi-task learning setting) 뒤의 일반화 기능 때문에 기존 방법에 비해 더 높은 시각적 품질의 이미지를 생성했다. 또한 제안된 단순 마스크 벡터를 사용하면 StarGAN은 서로 다른 도메인 레이블 세트를 가진 여러 데이터 세트를 활용할 수 있으므로 이들로부터 사용 가능한 모든 레이블을 처리할 수 있다. 우리는 사용자가 여러 도메인에 걸쳐 흥미로운 이미지 변환 응용 프로그램(image translation applications)을 개발할 수 있도록 하기 위한 우리의 작업을 희망한다.



## 3. Introduction

The task of image-to-image translation is to change a particular aspect of a given image to another, e.g., changing the facial expression of a person from smiling to frowning (see Fig. 1). This task has experienced significant improvements following the introduction of generative adversarial networks (GANs), with results ranging from changing hair color [9], reconstructing photos from edge maps [7], and changing the seasons of scenery images [33].

image-to-image translation의 작업은 주어진 이미지의 특정 측면(aspect)을 다른 측면으로 변경하는 것이다. 예를 들어, 사람의 얼굴 표정을 웃는 얼굴에서 미간을 찌푸리는 얼굴로 바꾸는 것이다(그림 1 참조). 이 작업은 generative adversarial networks(GANs)의 도입 이후 상당한 개선을 경험했으며, 헤어 색상 변경, edge maps의 사진 재구성, 풍경 이미지의 계절 변경 등의 결과가 나왔다.



Given training data from two different domains, these models learn to translate images from one domain to the other. We denote the terms attribute as a meaningful feature inherent in an image such as hair color, gender or age, and attribute value as a particular value of an attribute, e.g., black/blond/brown for hair color or male/female for gender. We further denote domain as a set of images sharing the same attribute value. For example, images of women can represent one domain while those of men represent another.

두 개의 서로 다른 도메인의 training data를 고려하여,  이 모델들은 한 도메인에서 다른 도메인으로 이미지를 변환하는 방법을 배운다. 우리는 속성이라는 용어를 머리색, 성별 또는 나이와 같은 이미지에 내재된 의미 있는 특징으로 나타내고 속성 값(예: 머리색에 대한 검은색/금발색/갈색 또는 성별에 대한 남성/여성)을 특성의 특정 값으로 나타낸다. 또한 도메인을 동일한 속성 값을 공유하는 이미지 집합으로 표시한다. 예를 들어, 여성의 이미지는 하나의 영역을 나타내는 반면 남성의 이미지는 다른 영역을 나타낼 수 있다.



Several image datasets come with a number of labeled attributes. For instance, the CelebA[19] dataset contains 40 labels related to facial attributes such as hair color, gender, and age, and the RaFD [13] dataset has 8 labels for facial expressions such as ‘happy’, ‘angry’ and ‘sad’. These settings enable us to perform more interesting tasks, namely multi-domain image-to-image translation, where we change images according to attributes from multiple domains. The first five columns in Fig. 1 show how a CelebA image can be translated according to any of the four domains, ‘blond hair’, ‘gender’, ‘aged’, and ‘pale skin’. We can further extend to training multiple domains from different datasets, such as jointly training CelebA and RaFD images to change a CelebA image’s facial expression using features learned by training on RaFD, as in the rightmost columns of Fig. 1.\

여러 이미지 데이터 세트에는 레이블이 지정된 여러 속성이 포함되어 있다. 예를 들어, CelebA 데이터 세트에는 머리색, 성별, 나이와 같은 얼굴 속성과 관련된 40개의 레이블이 포함되어 있으며, RaFD 데이터 세트에는 '행복', '분노', '슬픔'과 같은 8개의 얼굴 표정 레이블이 있다. 이러한 설정을 사용하면 여러 도메인의 속성에 따라 이미지를 변경하는 다중 도메인 image-to-image translation과 같은 보다 흥미로운 작업을 수행할 수 있다. 그림 1의 첫 다섯 칸은 CelebA 이미지가 '금발머리', '성별', '나이 든', '창백한 피부'의 네 가지 영역 중 하나에 따라 어떻게 번역될 수 있는지를 보여준다. 그림 1의 가장 오른쪽 열에서처럼 RaFD에 대한 training을 통해 학습된 기능을 사용하여 CelebA 이미지의 얼굴 표정을 변경하도록 CelebA 이미지와 RaFD 이미지를 공동으로 교육하는 등 다양한 데이터 세트의 여러 도메인을 training하는 것으로 확장할 수 있다.



However, existing models are both inefficient and ineffective in such multi-domain image translation tasks. Their inefficiency results from the fact that in order to learn all mappings among k domains, k(k−1) generators have to be trained. Fig. 2 (a) illustrates how twelve distinct generator networks have to be trained to translate images among four different domains. Meanwhile, they are ineffective that even though there exist global features that can be learned from images of all domains such as face shapes, each generator cannot fully utilize the entire training data and only can learn from two domains out of k. Failure to fully utilize training data is likely to limit the quality of generated images. Furthermore, they are incapable of jointly training domains from different datasets because each dataset is partially labeled, which we further discuss in Section 3.2.

그러나 기존 모델은 이러한 multi-domain image translation task에서 비효율적이고 효과적이지 않다. 이들의 비효율성은 k 도메인 간의 모든 매핑을 학습하기 위해 k(k-1)개의 generator를 훈련시켜야 한다는 사실에서 비롯된다. 그림 2(a)는 12개의 구별되는 generator networks가 4개의 서로 다른 도메인 사이에서 이미지를 변환하도록 훈련되어야 하는 방법을 보여준다. 한편, 얼굴 모양과 같은 모든 도메인의 이미지에서 학습할 수 있는 글로벌 기능이 있지만, 각 generator는 전체 training 데이터를 완전히 활용할 수 없고 두 개의 도메인에서만 학습할 수 있기 때문에 비효율적이다. training 데이터를 완전히 활용하지 못하면 생성된 이미지의 품질이 제한될 가능성이 있다. 또한, 각 데이터 세트에 부분적으로 레이블이 지정되기 때문에 서로 다른 데이터 세트의 도메인을 공동으로 훈련할 수 없다. 이 문제는 Section 3.2.에서 자세히 논의한다.

![캡처2](https://user-images.githubusercontent.com/59161837/105483967-fa395d00-5ced-11eb-96fa-8ae43da55ca6.PNG)

그림 2. cross-domain model과 제안된 모델인 StarGAN 간의 비교, (a) 다중 도메인을 처리하려면 cross-domain model이 모든 이미지 도메인 쌍에 대해 구축되어야 한다. (b) StarGAN은 단일 generator를 사용하여 여러 도메인 간의 매핑을 학습할 수 있다. 이 그림은 multi-domain을 연결하는 star topology를 나타낸다.



As a solution to such problems we propose StarGAN, a novel and scalable approach capable of learning mappings among multiple domains. As demonstrated in Fig. 2 (b), our model takes in training data of multiple domains, and learns the mappings between all available domains using only a single generator. The idea is simple. Instead of learning a fixed translation (e.g., black-to-blond hair), our generator takes in as inputs both image and domain information, and learns to flexibly translate the image into the corresponding domain. We use a label (e.g., binary or one-hot vector) to represent domain information. During training, we randomly generate a target domain label and train the model to flexibly translate an input image into the target domain. By doing so, we can control the domain label and translate the image into any desired domain at testing phase.

이러한 문제에 대한 해결책으로 우리는 여러 도메인 간의 매핑을 학습할 수 있는 새롭고 확장 가능한 접근 방식인 StarGAN을 제안한다. 그림 2 (b)에서 설명한 것처럼, 우리의 모델은 여러 도메인의 training 데이터를 채택하고 단일 generator만 사용하여 사용 가능한 모든 도메인 간의 매핑을 학습한다. 아이디어는 간단하다. 우리의 generator는 고정된 변환(예: 검은색에서 금색 머리카락)을 배우는 대신 이미지와  도메인 정보를 모두 입력하면서 이미지를 해당 도메인으로 유연하게 변환하는 방법을 학습한다. 우리는 도메인 정보를 나타내기 위해 레이블(예: 이진 또는 단일 핫 벡터)을 사용한다. training 중에 우리는 무작위로 target 도메인 레이블을 생성하고 입력 이미지를 target 도메인으로 유연하게 변환하도록 모델을 훈련시킨다. 이렇게 함으로써 우리는 도메인 레이블을 제어하고 테스트 단계에서 원하는 도메인으로 이미지를 변환할 수 있다.



We also introduce a simple but effective approach that enables joint training between domains of different datasets by adding a mask vector to the domain label. Our proposed method ensures that the model can ignore unknown labels and focus on the label provided by a particular dataset. In this manner, our model can perform well on tasks such as synthesizing facial expressions of CelebA images using features learned from RaFD, as shown in the rightmost columns of Fig. 1. As far as our knowledge goes, our work is the first to successfully perform multi-domain image translation across different datasets.

또한 도메인 레이블에 마스크 벡터를 추가하여 서로 다른 데이터 세트의 도메인 간에 공동 훈련을 가능하게 하는 간단하지만 효과적인 접근 방식을 소개한다. 우리가 제안한 방법은 모델이 알 수 없는 레이블을 무시하고 특정 데이터 세트에 의해 제공되는 레이블에 집중할 수 있도록 보장한다. 이러한 방식으로, 우리의 모델은 그림 1의 맨 오른쪽 열에 표시된 것처럼 RaFD에서 학습한 기능을 사용하여 CelebA 이미지의 얼굴 표정을 합성하는 것과 같은 작업에서 잘 수행할 수 있다. 우리가 아는 한, 우리의 작업은 서로 다른 데이터 세트에 걸쳐 다중 도메인 이미지 변환(multi-domain image tranlation)을 성공적으로 수행한 첫번째이다.



Overall, our contributions are as follows: 

• We propose StarGAN, a novel generative adversarial network that learns the mappings among multiple domains using only a single generator and a discriminator, training effectively from images of all domains. 

• We demonstrate how we can successfully learn multidomain image translation between multiple datasets by utilizing a mask vector method that enables StarGAN to control all available domain labels. 

• We provide both qualitative and quantitative results on facial attribute transfer and facial expression synthesis tasks using StarGAN, showing its superiority over baseline models.

전체적으로 우리의 contribution은 다음과 같다.

- 단일 generator와 discriminator만을 사용하여 여러 도메인 간의 매핑을 학습하는 새로운 generative adversarial network인 StarGAN을 제안하며, 모든 도메인의 이미지에서 효과적으로 훈련한다.
- StarGAN이 사용 가능한 모든 도메인 레이블을 제어할 수 있는 마스크 벡터 방법을 활용하여 여러 데이터 세트 간의 multi-domain image translation을 성공적으로 학습할 수 있는 방법을 보여준다.
- StarGAN을 사용한 얼굴 속성 전달 및 표정 합성 작업에 대한 정성적 및 정량적 결과를 모두 제공하여 기준 모델보다 우월함을 보여준다.



## 4. Related Work

Generative Adversarial Networks. Generative adversarial networks (GANs) [3] have shown remarkable results in various computer vision tasks such as image generation [6, 24, 32, 8], image translation [7, 9, 33], super-resolution imaging [14], and face image synthesis [10, 16, 26, 31]. A typical GAN model consists of two modules: a discriminator and a generator. The discriminator learns to distinguish between real and fake samples, while the generator learns to generate fake samples that are indistinguishable from real samples. Our approach also leverages the adversarial loss to make the generated images as realistic as possible.

Generative Adversarial Networks. Generative Adversarial Networks (GANs)는 이미지 생성, 이미지 변환, 초 해상도 이미지(super-resolution imaging) 및 얼굴 이미지 합성 같은 다양한 컴퓨터 비전 작업에서 주목할 만한 결과를 보여주었다. 일반적인 GAN 모델은 discriminator와 generator의 두 모듈로 구성된다. discriminator는 실제 샘플과 가짜 샘플을 구별하는 방법을 학습하는 반면, generator는 실제 샘플과 구분할 수 없는 가짜 샘플을 생성하는 방법을 학습한다. 우리의 접근 방식은 또한 adversarial loss를 활용하여 생성된 이미지를 최대한 사실적으로 만든다.



Conditional GANs. GAN-based conditional image generation has also been actively studied. Prior studies have provided both the discriminator and generator with class information in order to generate samples conditioned on the class [20, 21, 22]. Other recent approaches focused on generating particular images highly relevant to a given text description [25, 30]. The idea of conditional image generation has also been successfully applied to domain transfer [9, 28], superresolution imaging[14], and photo editing [2, 27]. In this paper, we propose a scalable GAN framework that can flexibly steer the image translation to various target domains, by providing conditional domain information.

Conditional GANs. GAN 기반 조건부 이미지 생성도 활발히 연구되었다. 이전 연구에서는 클래스에 조건화된 샘플을 생성하기 위해 discriminatro와 generator 모두에 클래스 정보를 제공했다. 다른 최근의 접근 방식은 주어진 텍스트 설명과 매우 관련성이 높은 특정 이미지를 생성하는 데 초점을 맞췄다. 조건부 이미지  생성 아이디어는 도메인 전송, 초 해상도 이미지 및 사진 편집에도 성공적으로 적용되었다. 본 논문에서는 조건부 도메인 정보를 제공하여 이미지 변환을 다양한 target 도메인으로 유연하게 조정할 수 있는 확장 가능한 GAN 프레임워크를 제안한다.



Image-to-Image Translation. Recent work have achieved impressive results in image-to-image translation [7, 9, 17, 33]. For instance, pix2pix [7] learns this task in a supervised manner using cGANs[20]. It combines an adversarial loss with a L1 loss, thus requires paired data samples. To alleviate the problem of obtaining data pairs, unpaired image-to-image translation frameworks [9, 17, 33] have been proposed. UNIT [17] combines variational autoencoders (VAEs) [12] with CoGAN [18], a GAN framework where two generators share weights to learn the joint distribution of images in cross domains. CycleGAN [33] and DiscoGAN [9] preserve key attributes between the input and the translated image by utilizing a cycle consistency loss. However, all these frameworks are only capable of learning the relations between two different domains at a time. Their approaches have limited scalability in handling multiple domains since different models should be trained for each pair of domains. Unlike the aforementioned approaches, our framework can learn the relations among multiple domains using only a single model.

Image-to-Image Translation. 최근 연구는 image-to-image translation에서 인상적인 결과를 달성했다. 예를 들어, pix2pix는 cGAN을 사용하여 supervised manner로 이 작업을 학습한다. adversarial loss와 L1 loss를 결합하므로 쌍으로 이루어진 데이터(paired data) 샘플이 필요하다. 데이터 쌍을 얻는 문제를 완화하기 위해, 페어링되지 않은 이미지 간 변환 프레임워크가 제안되었다. UNIT은 variational autoencoders(VAEs)를 두 개의 generator가 가중치를 공유하여 cross domain에서 이미지의 공동 분포를 학습하는 GAN 프레임워크인 CoGAN과 결합한다. CylcleGAN과 DiscoGAN은 cycle consistency loss를 활용하여 입력과 변환된 이미지 사이의 주요 속성을 보존한다. 그러나 이러한 모든 프레임워크는 한 번에 두 개의 서로 다른 도메인 사이의 관계를 학습할 수 있을 뿐이다. 각 도메인 쌍에 대해 서로 다른 모델을 교육해야 하므로 이들의 접근 방식은 여러 도메인을 처리하는 데 있어 확장성이 제한적이다. 앞서 언급한 접근 방식과 달리, 우리의 프레임워크는 단일 모델만 사용하여 여러 도메인 간의 관계를 학습할 수 있다.



## 5. Star Generative Adversarial Networks

We first describe our proposed StarGAN, a framework to address multi-domain image-to-image translation within a single dataset. Then, we discuss how StarGAN incorporates multiple datasets containing different label sets to flexibly perform image translations using any of these labels.

먼저 단일 데이터 세트 내에서 multi-domain image-to-image translation을 해결하기 위한 프레임워크인 제안된 StarGAN을 설명한다. 그런 다음 StarGAN이 다양한 레이블 세트를 포함하는 여러 데이터 세트를 통합하여 이러한 레이블 중 하나를 사용하여 이미지 변환을 유연하게 수행하는 방법에 대해 논의한다.

![캡처3](https://user-images.githubusercontent.com/59161837/105488805-9024b600-5cf5-11eb-8b1e-b3338826cc22.PNG)

그림 3. 두 개의 모듈로 구성된 StarGAN 개요, discriminator D와 generator G.

(a) D는 실제 이미지와 가짜 이미지를 구별하고 실제 이미지를 해당 도메인으로 분류하는 방법을 배운다.

(b) G는 이미지와 target domain label 모두 입력으로 사용하여 가짜 이미지를 생성한다.

(c) G는 원래의 도메인 라벨이 주어진 가짜 이미지에서 원본 이미지를 재구성하려고 한다.

(d) G는 실제 이미지와 구별되지 않고 D에 의해 대상 도메인으로 분류할 수 있는 이미지를 생성하려고 한다.



### 5.1. Multi-Domain Image-to-Image Translation

Our goal is to train a single generator G that learns mappings among multiple domains. To achieve this, we train G to translate an input image x into an output image y conditioned on the target domain label c, G(x, c) → y. We randomly generate the target domain label c so that G learns to flexibly translate the input image. We also introduce an auxiliary classifier [22] that allows a single discriminator to control multiple domains. That is, our discriminator produces probability distributions over both sources and domain labels, D : x → {Dsrc(x), Dcls(x)}. Fig. 3 illustrates the training process of our proposed approach.

우리의 목표는 여러 도메인 간의 매핑을 학습하는 단일 generator G를 훈련시키는 것이다. 이를 위해 G는 입력 이미지 x를 target domain label c, G(x,c) -> y에서 조건화된 출력 이미지로 변환하도록 훈련한다. 우리는 G가 입력 이미지를 유연하게 변환하는 방법을 학습하도록 target domain label c를 무작위로 생성한다. 또한 단일 discriminator가 여러 도메인을 제어할 수 있는 보조 분류기(auxiliary classifier)를 소개한다. 즉, 우리의 discriminator는 소스 sources and domain labels 모두에 대해 확률 분포를 생성한다. D : x ->{D_src(x), D_cls(x)}. 그림 3은 제안된 접근 방식의 training 과정을 보여준다.

![캡처4](https://user-images.githubusercontent.com/59161837/105489873-4b9a1a00-5cf7-11eb-8255-3995aa7f2c4b.PNG)

Adversarial Loss. 생성된 이미지를 실제 이미지와 구분할 수 없도록 하기 위해 adversarial loss를 채택한다.



where G generates an image G(x, c) conditioned on both the input image x and the target domain label c, while D tries to distinguish between real and fake images. In this paper, we refer to the term Dsrc(x) as a probability distribution over sources given by D. The generator G tries to minimize this objective, while the discriminator D tries to maximize it.

여기서 G는 입력 이미지 x와 target domain label c에서 조건화된 이미지 G(x,c)를 생성하는 반면, D는 실제 이미지와 가짜 이미지를 구별하려고 노력한다. 본 논문에서는 D가 제공한 소스에 대한 확률 분포로 D_src(x)라는 용어를 참조한다. generator G는 이 목표를 최소화하려고 하는 반면, discriminator D는 이를 최대화하려고 한다.

![캡처5](https://user-images.githubusercontent.com/59161837/105490182-c06d5400-5cf7-11eb-95d8-06426b1c7ad6.PNG)

Domain Classification Loss. 주어진 입력 이미지 x와 target domain label c의 경우, 우리의 목표는 x를 target domain c로 적절하게 분류되는 출력 이미지 y로 변환하는 것이다. 이 조건을 달성하기 위해 D 위에 보조 분류기(auxiliary classifier)를 추가하고 D와 G를 모두 최적화할 때 domain classification loss를 부과한다. 즉, D를 최적화하는 데 사용되는 실제 이미지의 domain classification loss와 G를 최적화하는 데 사용되는 가짜 이미지의 domain classification이라는 두 가지 용어로 목표를 분해한다. 구체적으로 전자는 (2)와 같이 정의된다. 여기서 용어 D_cls(c'|x)는 D가 계산한 도메인 레이블에 대한 확률 분포를 나타낸다. 이 목표를 최소화함으로써 D는 실제 이미지 x를 해당 원래 도메인 c'로 분류하는 방법을 배운다. 우리는 입력 입미지와 도메인 레이블 쌍(x,c')이 training 데이터에 의해 주어진다고 가정한다. 한편, 가짜 이미지의 domain classification에 대한 손실 함수는 (3)과 같이 정의된다. 즉, G는 target domain c로 분류될 수 있는 이미지를 생성하기 위해 이 목표를 최소화하려고 한다.

![캡처6](https://user-images.githubusercontent.com/59161837/105490903-d7f90c80-5cf8-11eb-89d6-98a22206a2df.PNG)

Reconstruction Loss, adversarial and classification losses를 최소화함으로써, G는 현실적이고 정확한 target domain으로 분류되는 이미지를 생성하도록 훈련된다. 그러나 손실(Eqs. (1) and (3))을 최소화한다고 해서 변환된 이미지가 입력의 도메인 관련 부분만 변경하면서 입력 이미지의 내용을 보존한다는 보장은 없다. 이 문제를 완화하기 위해, 우리는 (4)와 같이 정의된 generator에 cycle consistency loss를 적용한다. 여기서 G는 변환된 이미지 G(x, c)와 원본 도메인 레이블 c'를 입력으로 사용하여 원본 이미지 x를 재구성한다. 우리는 reconstruction loss로 L1 norm을 채택한다. 단일 generator를 두 번 사용하여, 먼저 원본 이미지를 target domain의 이미지로 변환한 다음 변환된 이미지에서 원본 이미지를 재구성한다.

![캡처7](https://user-images.githubusercontent.com/59161837/105491684-09260c80-5cfa-11eb-9d3f-35fc179e0d96.PNG)

Full Objective. 마지막으로, G와 D를 최적화하기 위한 목적 함수는 각각 다음과 같이 작성된다. 여기서 람다_cls 와 람다_crs는 adversarial loss에 비해 각각 domain classification과 reconstruction losses의 상대적 중요성(relative importance)을 제어하는 하이퍼 파라미터이다. 우리는 모든 실험에서 람다_cls = 1, 람다_rec = 10을 사용한다.



### 5.2. Training with Multiple Datasets

An important advantage of StarGAN is that it simultaneously incorporates multiple datasets containing different types of labels, so that StarGAN can control all the labels at the test phase. An issue when learning from multiple datasets, however, is that the label information is only partially known to each dataset. In the case of CelebA [19] and RaFD [13], while the former contains labels for attributes such as hair color and gender, it does not have any labels for facial expressions such as ‘happy’ and ‘angry’, and vice versa for the latter. This is problematic because the complete information on the label vector c 0 is required when reconstructing the input image x from the translated image G(x, c) (See Eq. (4)).

StarGAN의 중요한 장점은 다양한 유형의 레이블을 포함하는 여러 데이터 세트를 동시에 통합하여 테스트 단계에서 StarGAN이 모든 레이블을 제어할 수 있다는 것이다. 그러나 여러 데이터 세트에서 학습할 때 문제는 레이블 정보가 각 데이터 세트에 부분적으로만 알려져 있다는 것이다. CelebA와 RaFD의 경우, 전자는 머리카락 색과 성별과 같은 속성에 대한 레이블을 포함하고 있지만, '행복'과 '분노'와 같은 표정에 대한 라벨이 없으며, 후자의 경우 그 반대도 마찬가지이다. 변환된 이미지 G(x, c)에서 입력 이미지 x를 재구성할 때 레이블 벡터 c'에 대한 전체 정보가 필요하기 때문에 문제가 있다(Eq. (4) 참조).

![캡처8](https://user-images.githubusercontent.com/59161837/105493614-f7923400-5cfc-11eb-91c9-52c19bcf38fb.PNG)

이 문제를 완화하기 위해 StarGAN이 지정되지 않은 레이블을 무시하고 특정 데이터 세트에 의해 제공되는 명시적으로 알려진 레이블에 집중할 수 있는 마스크 벡터 m을 도입한다. StarGAN에서 우리는 n차원 원핫 벡터를 사용하여 m을 나타내며, n은 데이터 세트의 수이다. 또한, 우리는 통일된 버전의 레이블을 벡터로 정의한다. 

여기서 [·]는 연결을 의미하며, ci는 i번째 데이터 세트의 레이블에 대한 벡터를 나타낸다. 알려진 라벨 ci의 벡터는 binary attributes를 위한 binary vector 또는 categorical attributes를 위한 one-hot vector로 표현될 수 있다. 나머지 n-1 알 수 없는 라벨에 대해서는 0 값을 할당하기만 하면 된다. 우리의 실험에서, 우리는 CelebA와 RaFD 데이터 세트를 활용한다. 여기서 n은 2이다.



Training Strategy. When training StarGAN with multiple datasets, we use the domain label c˜ defined in Eq. (7) as input to the generator. By doing so, the generator learns to ignore the unspecified labels, which are zero vectors, and focus on the explicitly given label. The structure of the generator is exactly the same as in training with a single dataset, except for the dimension of the input label c˜. On the other hand, we extend the auxiliary classifier of the discriminator to generate probability distributions over labels for all datasets. Then, we train the model in a multi-task learning setting, where the discriminator tries to minimize only the classification error associated to the known label. For example, when training with images in CelebA, the discriminator minimizes only classification errors for labels related to CelebA attributes, and not facial expressions related to RaFD. Under these settings, by alternating between CelebA and RaFD the discriminator learns all of the discriminative features for both datasets, and the generator learns to control all the labels in both datasets.

Training Strategy. 여러 데이터 세트를 사용하여 StarGAN을 training할 때, 우리는 Eq.(7)에 정의된 도메인 레이블 c˜를 generator에 대한 입력으로 사용한다. 이를 통해 generator는 지정되지 않은 레이블(벡터 0)을 무시하고 명시적으로 지정된 레이블에 초점을 맞추는 방법을 학습한다. generator의 구조는 입력 라벨 c˜의 치수를 제외하고 단일 데이터 세트를 사용하는 훈련에서와 정확히 동일하다. 한편, 우리는 discriminator의 auxiliary classifier를 확장하여 모든 데이터 세트에 대한 레이블에 대한 확률 분포를 생성한다. 그런 다음, 우리는 식별자가 알려진 라벨과 관련된 classification error만 최소화하려고 하는 multi-task learning setting에서 모델을 훈련시킨다. 예를 들어 CelebA에서 이미지로 훈련할 때 discriminator는 CelebA 속성과 관련된 레이블에 대한 classification error만 최소화하고 RaFD와 관련된 표정은 최소화하지 않는다. 이러한 설정에서 CelebA와 RaFD를 번갈아 사용함으로써 discriminator는 두 데이터 세트에 대한 모든 discriminative features를 학습하고 generator는 두 데이터 세트의 모든 레이블을 제어하는 방법을 학습한다.



## 6. Implementation

![캡처9](https://user-images.githubusercontent.com/59161837/105495215-3f19bf80-5cff-11eb-930f-82556fbe931c.PNG)

training process를 안정화하고 고품질 이미지를 생성하기 위해, 우리는 Eq. (1)을 (8)과 같이 정의된 gradient penalty로 Wassersteing GAN objective로 대체한다.  여기서 x^은 실제 이미지와 생성된 이미지 쌍 사이의 직선을 따라 균일하게 샘플링된다. 우리는 모든 실험에 람다_gp = 10을 사용한다.

Network Architecture. CycleGAN에서 채택된 StarGAN은 다운샘플링의 경우 2개의 stride size를 가진 convolutional layers와 업샘플링의 경우 6개의 residual blocks, 2개의 stride size를 가진 2개의 convolutional layers로 구성된 generator network를 가지고 있다. 우리는 generator에 instance normalization을 사용하지만 discriminator에 대해서는 normalization이 없다. 우리는 local image patches가 실제인지 가짜인지를 분류하는 판별기 네트워크에 대해 PatchGANs을 활용한다. network architecture에 대한 자세한 내용은 appendix (Section 7.2)를 참조하라.



## 7.  Experiments

### 7.1. Baseline Models

### 7.2. Datasets

### 7.3. Training

### 7.4. Experimental Results on CelebA

### 7.5. Experimental Results on RaFD

### 7.6. Experimental Results on CelebA+RaFD

