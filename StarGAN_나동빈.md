# 나동빈님의 StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)

- Link: https://www.youtube.com/watch?v=-r9M4Cj9o_8&t=1908s

![그림1](https://user-images.githubusercontent.com/59161837/105186668-4ca33e00-5b75-11eb-92ec-642e09b948bf.PNG)

- 도메인이 무엇인가?: 이미지 변환이란 어떤 이미지를 특정한 특징을 가진 이미지로 변환하는 것을 말한다. 예를 들어 무표정인 사람의 모습을 웃는 모습으로 바꾼다거나 혹은 성별을 바꾸는 등 다양한 특징을 가지게 변환시키는 작업을 '이미지 변환', 변환할 때 적용하는 특징을 '도메인'이라고 부른다. 출처: https://medium.com/curg/stargan-%EB%8B%A8%EC%9D%BC-%EB%AA%A8%EB%8D%B8%EB%A1%9C-%EB%8B%A4%EC%A4%91-%EB%8F%84%EB%A9%94%EC%9D%B8-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B3%80%ED%99%98%EA%B8%B0-%EB%A7%8C%EB%93%A4%EA%B8%B0-3b0fbdec121d
  - 이 그림에서는 Black hair, Blond hair, Gender, Aged가 도메인이다.
- 다른 네트워크 같은 경우에는 특정 도메인에서 다른 여러 개의 도메인으로 바꾸려고 하면, 여러 개의 네트워크가 필요하거나 네트워크를 중첩해서 사용하도록 만들어야지 가능했다. 
  - 다중 도메인에서의 효율적인 image-to-image translation 네트워크, 하나의 네트워크를 제안한 것이 StarGAN의 Contribution이다.

![캡처2](https://user-images.githubusercontent.com/59161837/105188112-e4555c00-5b76-11eb-8435-1f1a5b75627b.PNG)



![캡처3](https://user-images.githubusercontent.com/59161837/105188283-136bcd80-5b77-11eb-9950-8b45abaefcc3.PNG)

![캡처4](https://user-images.githubusercontent.com/59161837/105188484-4ca43d80-5b77-11eb-8f93-8e3f8bde388d.PNG)

![캡처5](https://user-images.githubusercontent.com/59161837/105188961-cccaa300-5b77-11eb-8c49-3139349469fc.PNG)

- 기존 GAN에서 조건부 형태로 y가 들어간 것을 볼 수 있다.
  - MNIST 같은 경우: y는 0~9까지의 숫자 즉, label이 된다. 
    - Generator는 특정 class에 맞는 이미지를 만들 수 있도록 학습이 된다. 따라서 특정 조건(class)에 맞는 이미지를 만들도록 할 수 있다.

![캡처6](https://user-images.githubusercontent.com/59161837/105189748-a0fbed00-5b78-11eb-8829-1aaa1599dbaf.PNG)

![캡처7](https://user-images.githubusercontent.com/59161837/105189940-d99bc680-5b78-11eb-97de-e77678342412.PNG)

- Pix2pix는 초기에 나온 Image-to-Image Translation 기법을 위한 아키텍처 중 하나다.
  - I2I translation은 주어진 이미지가 있을 때, 특정한 aspect(양상, 특징)을 골라서 다른 특징으로 바꿀 수 있도록 만드는 것이다.
    - 예를 들면, 얼굴이 들어왔을 때, 눈썹을 골라서 눈썹이 없는 얼굴로 만드는 것.
- cGAN에서 MNIST 데이터의 class를 조건으로 했다면, Pix2pix는 이미지를 조건으로 입력한다.

![캡처8](https://user-images.githubusercontent.com/59161837/105190879-d35a1a00-5b79-11eb-84c6-73d6d03c324f.PNG)

- Pix2pix는 실존하는 이미지(yi)와 손그림 이미지(xi)가 한 쌍으로 묶여서 학습이 진행된다. 따라서 데이터셋을 구성하기 어려울 수도 있다.
  - 한 쌍으로 묶이지 않은(Unpaired) 데이터셋은 어떡하지? -> CycleGAN을 이용하자.

![캡처9](https://user-images.githubusercontent.com/59161837/105191691-b5d98000-5b7a-11eb-8881-6b87ee75b759.PNG)

- CycleGAN의 문제: 얼룩말 이미지가 잘 뽑히면 계속 얼룩말만 만든다.

![캡처10](https://user-images.githubusercontent.com/59161837/105192222-38623f80-5b7b-11eb-834c-302dfa3ffde8.PNG)



- CycleGAN의 문제를 해결하기 위해 생성자가 만든 이미지를 다시 원본 이미지 x로 재구성될 수 있는 형태로 만들어지도록 한다.
  - 이를 통해 원본 이미지의 content는 보존하고 도메인과 관련한 특징을 바꿀 수 있다.
    - 이 과정이 Cycle을 도는 것 같아서 Cycle-consistency loss라고 부른다.
- 이렇게 서로 다른 input image를 무시하고 얼룩말만 뽑는게 아니라, 각각의 input image의 컨텐츠에 대한 정보는 유지하면서, 특정 도메인으로 그 특성값만 바꾸는 것이 CycleGAN이다.

![캡처11](https://user-images.githubusercontent.com/59161837/105194742-40bb7a00-5b7d-11eb-8b7e-0128cb0e5bc8.PNG)



- 1-Lipshichtz 조건?: 유계폐구간이 아닌 곳에서 균등연속함수를 판단하는 방법의 하나로 Lipshichtz(립쉬츠) 조건이 있다. 임의의 두 점 x, u에 대해서 두 점의 차의 상수 K배가 두 점에서의 함숫값의 차보다 크거나 같다면, 이를 립쉬츠 조건 또는 립쉬츠 함수라 부른다. 출처: https://m.blog.naver.com/PostView.nhn?blogId=oyuniee&logNo=221234472441&proxyReferer=https:%2F%2Fwww.google.com%2F

  ![캡처12](https://user-images.githubusercontent.com/59161837/105196158-afe59e00-5b7e-11eb-86e9-8bad7450d96b.PNG)

  위 그림은 립쉬츠 연속성을 가장 직관적으로 잘 표현한 이미지다. 이미지를 보면 파란색 함수의 접선이 모두 분홍색 영역에 있음을 머리속에 그려볼 수 있다. 

  - Lipshichtz continuity, 어디다 써먹어?: Gradient descent 같은 최적화 방법론에서 비용함수가 립쉬츠 연속성을 보인다면 Gradient exploding 같은 문제를 미연에 방지할 수 있다. 출처: https://light-tree.tistory.com/188

- weight clipping?

  ![캡처13](https://user-images.githubusercontent.com/59161837/105273372-7e042400-5bde-11eb-9991-8c4d9d476ea0.PNG)

  위 내용은 WGAN의 최종적인 학습 알고리즘이다. 먼저 n critic번 만큼 critic을 학습시키는 부분이 보이는데, Pr과 p(z) (Ptheta 역할)를 미니 배치만큼 샘플링한 후에, critic의 loss function을 이용하여 parameter w(즉, 함수 f)를 update시킨다. 여기서 update 후 clip(w, -c, c)라는 부분이 있는데, Lipshichtz조건을 만족하도록 parameter w가 [-c, c]공간 안쪽에 존재하도록 강제하는 것이다. 이를 weight clipping이라고 한다. 이는 WGAN의 한계점이라고 할 수 있는데, 실험 결과 clipping parameter c가 크면 limit(c나 -c)까지 도달하는 시간이 오래 걸리기 때문에, optimal 지점까지 학습하는 데 시간이 오래 걸렸다고 한다. 반면 c가 작으면, gradient vanishing 문제가 발생하였다고 한다. 이미 간결하고 성능이 좋기 때문에 사용하였지만, 이후의 발전된 방법으로  Lipshichtz조건을 만족시키는 것은 다른 학자들에게 맡긴다라고 쓰여있다. 

  ![캡처14](https://user-images.githubusercontent.com/59161837/105273821-5d889980-5bdf-11eb-822d-2658a4cd0a16.PNG)

  위 그림은 batch normalization 없이 실험한 결과인데, clipping parameter c에 매우 민감하게 gradient가 변화함을 볼 수 있다고 한다. 또한 clipping의 문제점은, 이것이 regularizer로써 작용하여 함수 f의 capacity를 줄인다는 것도 있다고 한다. 때문에 이러한 점을 보완하기 위해 만들어진 것이 gradient penalty(그래프에서 파란색 선)를 준  WGAN-GP다. 출처: https://ahjeong.tistory.com/7

- gradient penalty?

  - WGAN-GP: Improved WGAN이다. WGAN이 k-Lipshichtz constraints를 만족시키기 위해 단순히 clipping을 수행하는데, 이것이 학습을 방해하는 요인으로 작용할 수 있다. WGAN-GP에서는 gradient penalty라는 것을 목적함수에 추가하여 이를 해결하였고, 학습 안정성을 데이터셋뿐만 아니라 모델 architecture에 대해서도 얻어냈다.

  - WGAN은 clipping을 통해 Lipshichtz 함수 제약을 해결하긴 했지만, 이는 예상치 못한 결과를 초래할 수 있다. 

    - WGAN 논문에서 인용: 만약 clipping parameter(c)가 너무 크다면, 어떤 weights든 그 한계에 다다르기까지 오랜 시간이 걸릴 것이며, 따라서 D가 최적화되기까지 오랜 시간이 걸린다. 반대로 c가 너무 작다면, 레이어가 크거나 BatchNorm을 쓰지 않는다면 쉽게 vanishing gradient 문제가 생길 수 있다.

    clipping은 단순하지만 문제를 발생시킬 수 있다. 특히 c가 잘 정해지지 않았다면 품질이 낮은 이미지를 생성하고 수렴하지 않을 수 있다. 모델의 성능은 이 c에 매우 민감하다.

  - weight clipping은 가중치를 정규화하는 효과를 갖는다. 이는 모델 f의 어떤 한계치를 설정하는 것과 같다. 그래서 이 논문(WGAN-GP)에서는 gradient penalty라는 것을 D의 목적함수에 추가해 이 한계를 극복하고자 한다. (G의 목적함수는 건드리지 않는 듯 하다.)

    ![캡처15](https://user-images.githubusercontent.com/59161837/105275029-c113c680-5be1-11eb-9b25-57ff0c11fac1.PNG)

    출처: https://greeksharifa.github.io/generative%20model/2019/03/20/advanced-GANs/

- Inception Score?: Inception Score(IS)는 GAN의 성능을 측정하기 위해 다음 두 가지 기준을 고려한다. 

  1. 생성된 이미지의 quality (진짜 같은 이미지가 만들어지는지)
  2. diversity (다양한 이미지가 만들어지는지)

  엔트로피는 randomness로 볼 수 있는데, 확률 변수 x가 뻔하게 예측가능하다면 엔트로피가 낮다고 볼 수 있다. GAN에서는 조건부 확률 P(y|x)가 예측 가능성이 높기를(생성된 이미지의 클래스를 예측하기 쉬워야 함) 원하고 이는 낮은 엔트로피를 가져야 함을 알 수 있다. 

  - 여기서 x는 생성된 이미지, y는 label이다.
  - IS에서는 생성된 이미지의 클래스를 예측할 때 pre-train된 inception network를 사용한다.

  ![캡처16](https://user-images.githubusercontent.com/59161837/105275577-05ec2d00-5be3-11eb-889e-acb12aad49d6.PNG)

  출처: https://cyc1am3n.github.io/2020/03/01/is_fid.html

![캡처17](https://user-images.githubusercontent.com/59161837/105285612-1529a580-5bf8-11eb-8fd5-be143149efed.PNG)

![캡처18](https://user-images.githubusercontent.com/59161837/105285734-591caa80-5bf8-11eb-9e89-ba4d51bf929c.PNG)

![캡처19](https://user-images.githubusercontent.com/59161837/105285903-a9940800-5bf8-11eb-8478-b7766d5073c3.PNG)

- Adversarial loss: 생성 이미지가 있을 법한 이미지로 보일 수 있게 하기 위한 loss.
  - 기본 GAN의 loss와 유사하게 생겼다.
  - G(x,c)에서 c는 특정 도메인을 의미한다.
  - G같은 경우는 real image로 분류될 수 있도록 학습을 진행하고, D같은 경우는 fake image로 분류될 수 있도록 학습을 진행한다.
- 최종 목적 함수: L_D같은 경우 Adversarial loss에 마이너스를 붙임으로써, maximize하는 것이다. L_G같은 경우 Adversarial loss를 minimise하는 것이다.
- Domain classification: c로 표현되는 condition이 도메인 정보라고 보면 된다.
  - L_cls_f는 fake image에서 generator를 위해 사용되는 것으로, D_cls(c|G(x,c))에서 G(x,c)는 특정 도메인으로 생성한 이미지이고, 그 생성한 미지가 타겟 도메인(c)으로 분류될 수 있는 형태로 학습이 진행된다.
  - L_cls_r은 real image의 상황에서, D_cls(c'|x)의 x는 real image이고, c'는 real image에 대한 도메인 값이다. 즉, 원래 이미지를 원래 이미지의 도메인 값으로 분류할 수 있도록 학습된다.
- Reconstruction: generator가 만든 이미지가 traslation을 진행할 때 원본 이미지 형태(어떤 content, identity라고 표현)는 유지되면서 도메인 정보만 바뀔 수 있도록 cycle loss를 이용한다. 그리고 L1 loss를 이용하여 원본 이미지와 reconstruction한 이미지가 유사한 형태가 될 수 있도록 loss 값을 구성해서 쓴다.
- 최종 목적 함수: L_D 같은 경우 real image에 대한 classification 결과를 뱉을 수 있도록 하는 형태여야 하므로, L_cls_r이 들어가 있다. L_G 또한 L_cls_f와 L_rec를 가져가서 사용하고 있다.
- 이러한 목점 함수에 WGAN-GP까지 적용을 해서 만들어진 목적 함수를 이용해서 training을 진행하는 것이 특징이다.

![캡처20](https://user-images.githubusercontent.com/59161837/105324537-e66af980-5c0e-11eb-8f57-17f3633e588b.PNG)

- SNG: single network: RaFD는 데이터셋의 크기가 크지 않다. 따라서 결과가 그렇게 좋지 않다. JNT: CelebA와 RaFD를 둘 다 활용해서 RaFD label을 이용하면 더 잘 학습하는 것을 볼 수 있다.
  - StarGAN은 Mask vector를 통해 여러 개의 Multiple dataset에서 학습을 할 수 있고, 이로 인해 성능을 높일 수 있다.

![캡처21](https://user-images.githubusercontent.com/59161837/105325358-d99ad580-5c0f-11eb-9c44-cc0e0aa0bb70.PNG)

- (a)에서 discriminator는 image가 들어 왔을 때, real image인지 판별을 한다. 만약 real image면 CelebA label이 무엇인지도 판별한다. 만약 fake image면 label은 판별하지 않는다.
- (b)에서 generator는 input image와 target domain label을 받아서 image를 생성한다.
- (c)에서 generator는 output image와 original domain label을 받아서 Reconstructed image를 만든다.
