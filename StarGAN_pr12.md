# PR12-PR152: StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

- Link: https://www.youtube.com/watch?v=i3-rTEFpyv0&feature=youtu.be

![캡처](https://user-images.githubusercontent.com/59161837/105566209-41673280-5d6e-11eb-858b-f55548769c30.PNG)

- 데이터가 Paired 할 수도 있고, Unpaired 할 수도 있다.
  - Paired 한 데이터보다 Unpaired 한 데이터를 구하는 게 더 쉽다.
    - Unpaired 한 데이터로 된 문제를 풀고 있다. (발표자는 Unsupervised하다고 표현하고 있다.)
- Multi-Domain Image-to-Image Translations
  - 이미지를 '행복' -> '슬픔'으로만 바꾸는 게 아니라, '슬픔', '화남', '놀람' 등 여러 개의 도메인(발표자는 attribution으로 표현하고 있다.)을 동시에 바꾼다

![캡처2](https://user-images.githubusercontent.com/59161837/105566351-14ffe600-5d6f-11eb-9beb-258667e1fd4e.PNG)

- 제일 아래 그림을 보면, 사진을 여러 가지 화가의 그림으로 바꾼 것이다.
  - CycleGAN에서는 각각 다른 모델을 썼다. 사진을 Monet 그림으로 바꾸는 데 한 모델, 사진을 Van Gogh 그림으로 바꾸는 데 한 모델, 사진을 Cezanne 그림으로 바꾸는 데 한 모델, 사진을 Ukiyo-e 그림으로 바꾸는 데 한 모델, 총 4개의 모델이 필요하다.
    - StarGAN 논문은 이렇게 여러 가지 그림으로 바꾸는 데 단 하나의 모델로 바꾼다는 게 차이점이다.

![캡처3](https://user-images.githubusercontent.com/59161837/105566449-a7a08500-5d6f-11eb-9b17-bfaf60250690.PNG)

- X라는 도메인과 Y라는 도메인을 translation하는게 CycleGAN의 task다.
  - G는 X->Y로 바꾸는 인코더, F는 Y->X로 바꾸는 인코더, Dx는 X가 real인지 fake인지 맞추는 판별기, Dy는 Y가 real인지 fake인지 맞추는 판별기
- CycleGAN에서 중요한 것은 cycle-consistency loss이다. 만약 이것이 없으면 (두 번째 그림에서는) x와 관계없이 Dy만을 속일 수 있는 이미지를 생성할 것이다. 마찬가지로 (세 번째 그림에서는) y와 관계없이 Dx만을 속일 수 있는 이미지를 생성할 것이다.

![캡처4](https://user-images.githubusercontent.com/59161837/105566624-7aa0a200-5d70-11eb-81a0-420153c27c02.PNG)

- loss function은 두 개로 이루어져 있다.
  - Adversarial Loss / Cycle-consistency Loss
- Adversarial Loss를 보면 Dy가 y(real image) 인지를 맞추고, Dy가 x에서 시작되어 translation된 G(x)(fake image) 인지를 맞추는 방식으로 계산된다.
- Cycle-consistency Loss를 보면 G에 갔다가 F에 가서 돌아온 F(G(x))와 x를 비교하는, F에 갔다가 G에 가서 돌아온 G(F(y))와 y를 비교하는 방식으로 계산된다.

- Full Objective: X에서 출발하는 Adversarial Loss와 Y에서 출발하는 Adversarial Loss와 Cycle-consistency Loss에 람다라는 하이퍼 파라미터를 붙여서 만들었다.

![캡처5](https://user-images.githubusercontent.com/59161837/105566806-7e80f400-5d71-11eb-89f5-89da38c7fe0e.PNG)

- 문제 1: 하나의 CycleGAN은 2개의 인코더가 필요하다.
  - 1 도메인과 2 도메인을 왔다 갔다 하려면 G21과 G12가 필요하다.
    - 도메인 갯수가 n개면 n*(n-1)개의 모델이 필요하다.
- 문제 2: 1 도메인과 2 도메인을 training 시킬 때 쓰인 data가 1 도메인과 4 도메인을 training 시킬 때 쓰일 수가 없다.
- StarGAN
  - 문제 1: 하나의 인코더로 다 만든다.
  - 문제 2: 모든 data를 다 써서 training 시킬 수 있다.
- StarGAN의 단점: generator를 하나만 쓰기 때문에 1, 2, 3, 4, 5번 이미지의 크기가 다 같아야 한다. CycleGAN은 network가 다 다르기 때문에 이미지가 같을 필요는 없다.

![캡처6](https://user-images.githubusercontent.com/59161837/105566964-4ded8a00-5d72-11eb-8537-125eb874fd49.PNG)

- StarGAN은 unified 모델이기 때문에 하나의 generator와 하나의 discriminator를 사용한다.
- (a)를 보면, real image와 fake image가 discriminator에 들어간다. original GAN에서는 이 이미지가 real인지 fake인지만 구별한다. StarGAN에서는 real image라면, Domain classification까지 한다.
- (b)를 보면, conditional GAN과 비슷한데, input image와 target domain을 넣어준다. fake image를 생성하여 (d)로 보내서 discriminator에 넣는다. 그럼 (a)의 과정이 된다. 그리고 target domain과 domain classification이 맞는지 확인하여 generator에 피드백을 준다.
- (c)를 보면, fake image와 original domain을 conditional로 넣어줘서 다시 돌아올 수 있는지 Reconstructed image를 만들어서 input image와 비교를 한다.

![캡처7](https://user-images.githubusercontent.com/59161837/105567145-978aa480-5d73-11eb-9136-36cc358e8aa5.PNG)

- 문제: CelebA와 RaFD 데이터를 사용했는데, 각 데이터가 가지고 있는 label이 다르다.
  - CelebA는 hair color, gender, etc.
  - RaFD는 happy, angry, etc.
  - 각각의 이미지가 데이터를 여러 개 같이 training 할 때는 partial information of the labels 한다는 것이 문제다.
    - StarGAN은 mask vector(m)을 사용해서 해결한다고 한다.
- mask vector: 이 이미지가 어떤 label을 가지고 있는지 network에 전달해준다.

![캡처8](https://user-images.githubusercontent.com/59161837/105567391-de2cce80-5d74-11eb-9e51-4dd512f9d1ff.PNG)

- 하나의 generator만을 사용하기 때문에 flexibility가 떨어진다고 생각한다. 여기서도 CelebA와 RaFD 이미지 사이즈가 다르다. 그래서 이걸 128 X 128로 크롭해서 쓰는데, 이게 단점인 것 같다.
- Multiple dataset가 된다고 했는데, 여기서는 2가지 데이터 세트만 이용했기 때문에, 3개 이상의 데이터를 넣었을 때 어떻게 될지는 모른다. 이게 단점인 것 같다.
