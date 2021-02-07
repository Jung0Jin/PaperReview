# StarGAN_질문리스트

## 1. 논문에서 제시된 두 dataset는 각각을 domain이라고 할 수 있는가?

![캡처](https://user-images.githubusercontent.com/59161837/105567801-1550af00-5d78-11eb-92b9-b3ec5b856444.PNG)

- Domain이란 무엇인가?: 이미지 변환이란 어떤 이미지를 특정한 특징을 가진 이미지로 변환하는 것을 말한다. 예를 들어 무표정인 사람의 모습을 웃는 모습으로 바꾼다거나 혹은 성별을 바꾸는 등 다양한 특징을 가지게 변환시키는 작업을 '이미지 변환', 변환할 때 적용하는 특징을 '도메인'이라고 부른다. 출처: https://medium.com/curg/stargan-%EB%8B%A8%EC%9D%BC-%EB%AA%A8%EB%8D%B8%EB%A1%9C-%EB%8B%A4%EC%A4%91-%EB%8F%84%EB%A9%94%EC%9D%B8-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B3%80%ED%99%98%EA%B8%B0-%EB%A7%8C%EB%93%A4%EA%B8%B0-3b0fbdec121d
- StarGAN 논문에서 해결하고자 한 문제와 기여는 다음과 같다.
  - 최근 연구에서 사용되는 도메인 - 변환 기법은 두 개 이상의 도메인을 학습할 때 Scalability와 Robustness가 부족하다.
  - StarGAN은 하나의 통합된 모델을 사용해서 다중 - 도메인과 다양한 데이터셋을 하나의 네트워크에서 학습할 수 있게 해준다.
    - 따라서 dataset을 domain이라고 하기 보단, dataset는 2가지이고(CelebA, RaFD) domain은 각각 [Blond hair, Gender, Aged, Pale Skin], [Angry, Happy, Fearful]로 표현하는게 맞는 것으로 보인다. 출처: https://medium.com/curg/stargan-%EB%8B%A8%EC%9D%BC-%EB%AA%A8%EB%8D%B8%EB%A1%9C-%EB%8B%A4%EC%A4%91-%EB%8F%84%EB%A9%94%EC%9D%B8-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B3%80%ED%99%98%EA%B8%B0-%EB%A7%8C%EB%93%A4%EA%B8%B0-3b0fbdec121d

## 2. G를 학습하는 과정에서, 만약 D가 G(x, c)를 fake라고 분류하면 c에 대한 학습을 할 수 없는 것 아닌가?
 
![캡처2](https://user-images.githubusercontent.com/59161837/105568057-f3582c00-5d79-11eb-8a57-99c232d70a7d.PNG)

![캡처3](https://user-images.githubusercontent.com/59161837/105568297-96f60c00-5d7b-11eb-8f81-91f11adace87.PNG)

- G를 학습하는 과정에서, 만약 D가 G(x, c)를 fake라고 분류하면 c에 대한 학습을 할 수 없는 것 아닌가?

  1. G가 c에 대한 학습을 할 수 없는 것 아닌가?

     L_cls_f는 fake image에서 generator을 학습시키기 위해 사용되는 loss이다. G(x, c)는 특정 도메인 c로 생성된 이미지이고, 그 생성된 이미지가 타겟 도메인 c로 분류될 수 있는 형태로 학습이 진행되므로, fake라고 분류하는 것과 상관없이 G는 학습을 진행할 수 있다.

  2. D가 c에 대한 학습을 할 수 없는 것 아닌가?

     L_cls_r은 real image에서 x(real image)를 이미지의 도메인 값 c'(real image에 대한 도메인 값)으로 분류할 수 있도록 학습되기 때문에, fake라고 분류하는 것과 상관없이 D는 학습을 진행할 수 있다.
 


## 3. 이 방법이 input image의 컨텐츠를 보존할 수 있다는 것이랑 무슨 관련이 있는지?

![캡처4](https://user-images.githubusercontent.com/59161837/105568426-83977080-5d7c-11eb-9136-3aa357104c53.PNG)

![캡처5](https://user-images.githubusercontent.com/59161837/105568449-a88be380-5d7c-11eb-99fb-e805cd08066d.PNG)

- CycleGAN에서 별도의 제약 조건 없이 단순히 입력 이미지 x의 일부 특성을 타겟 도메인 y의 특성으로 바꾸고자 한다면 어떤 입력이든 상관없이 특정한 도메인에 해당하는 하나의 이미지만 제시하게 될 수도 있다
  - Discriminator가 통과시켜주는 걸로만 계속 생성을 진행하게 된다
    - 그럼 입력 이미지의 컨텐츠가 없어진다.

![캡처6](https://user-images.githubusercontent.com/59161837/105568475-edb01580-5d7c-11eb-9613-ff29b6ba25d4.PNG)

- CycleGAN은 G(x)가 다시 원본 이미지 x로 재구성(reconstruct)될 수 있는 형태로 만들어지도록 한다.
  - 이를 통해 원본 이미지의 content는 보존하면서 도메인과 관련한 특징을 바꿀 수 있는 것이다.
    - StarGAN에서도 이 방법(cycle consistency)을 사용하여 input image의 컨텐츠를 보존한다 

## 4. 두 domain을 동시에 설정하여 생성할 수는 없는가?

![캡처7](https://user-images.githubusercontent.com/59161837/105568539-7dee5a80-5d7d-11eb-8ed0-b99b0f929508.PNG)

![캡처8](https://user-images.githubusercontent.com/59161837/105568553-ad04cc00-5d7d-11eb-9a69-b757603ee9dd.PNG)

- mask vector를 사용하는 이유: 이 이미지가 어떤 label을 가지고 있는지 network에 전달해주기 위해서 
  - 왜?: CelebA와 RaFD 데이터를 사용했는데, 각 데이터가 가지고 있는 label이 다르기 때문에, 각각의 이미지가 partial information of the labels 하다.
    - 따라서 두 domain은(사실 두 도메인이 아니라 두 데이터셋으로 표현하는 게 맞을 것 같다) 동시에 설정할 수 없다고 보인다. 
    - 동시에 설정하지말고 CelebA의 Black + Brown + Male에 해당하는 마스크 벡터를 통과시켜 생성된 이미지를 RaFD의 happy에 해당하는 마스크 벡터를 통과시키면 될 것 같다.
