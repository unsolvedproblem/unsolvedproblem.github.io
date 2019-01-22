---
layout: post
title:  "[쉽게읽는 머신러닝-라온피플] 1. 머신러닝 framework"
date:   2019-01-22
category: laon
tags: laon
author: Polar B, 백승열
comments: true
---
<br><br>
앞으로 우리팀에서 스터디를 통해 공부했던 내용들을 정리하여 각 주마다 연재를 할 계획입니다. 저희가 처음 머신러닝을 접해서 차례대로 공부한 내용들을 올리는 것이니 머신러닝을 처음 시작하시는 분들께서 보신다면 도움이 될것입니다.
<br><br><br>
그럼, 시작하겠습니다!
<br><br><br>

## 1. 학습의 정의
<br>
- 학습은 <u>사용한 데이터</u>로부터 <u>중요한 특징</u>을 끄집어내고, 그 특징으로부터 <u>일반화</u>를 거쳐 <u>예측할 수 잇는 능력</u>을 세우는 것을 말합니다.
<br><br>
- 학습에 사용하는 데이터의 '양'만 중요한 것이 아니라, '질'도 매우 중요합니다.
<br><br>
- 학습에는 오류가 있을 수 있으며, 오류가 발생하더라도 재학습을 통해 학습의 질을 개선할 수 있습니다.
<br><br>

## 2. 학습의 미시적인 수준에서의 의미
<br>
#### 인간의 두뇌
<br>
사람들은 컴퓨터를 어떻게 학습시킬지 생각하다가 사람의 학습방법을 모방하기로 합니다. 여기서는 미시적 수준 (즉, 세포 수준)에서 학습에 관여하는 각 세포들의 역할을 알아보겠습니다.
<br><br>
![인간의 두뇌](/assets/images/Laon/week1-1-1.png){: width="70%" height="auto" .image-center}
<br><br>
 - 세포체(Soma) :
 <br><br>
 세포로 들어온 신호의 수준을 판단하여 일정 수준 이하의 신호는 무시하고 일정 수준을 넘게 되면 받은 신호에 응답하여 다음 세포로 신호를 전송합니다. 즉, 세포체는 활성함수(activation function)기능을 담당하는 것이죠.
<br><br>
 - 수상돌기(Dendrite) :
 <br><br>
 세포로 전달되는 신호를 받아들이는 부분입니다.
<br><br>
 - 축삭돌기(Axon) :
 <br><br>
 세포에서 다른 세포로 신호를 전달하는 부분입니다.
<br><br>
 - 시냅스(Synapse) :
 <br><br>
 수상돌기와 축삭돌기 사이에 있는 부분으로 신호 전달의 세기(weight)를 담당하며 학습에 있어 매우 중요한 역할을 합니다. 즉, 어떤 세포의 축살돌기로부터 전달된 신호를 다음 세포로 보낼 때의 가중치를 결정하며, 0과 1사이의 범위의 숫자를 곱해주는 것으로 이해하면 됩니다.
 <br><br><br>
 _(0이면 전달되는 모든 신호가 무시되고, 그 값이 1쪽으로 갈수록 신호의 손실이 없이 그대로 전달이 되며, 학습을 하게되면 이 시냅스의 세기가 바뀌게 됩니다.<br>
 각 세포로 연결되는 수많은 시냅스의 세기(weight)가 학습의도에 따라 변하게 되는 것이죠.)_
<br><br><br>

#### 틀린그림찾기
<br>
그림에서 틀린 부분을 한번 찾아 볼까요?!
<br><br>
![틀린그림찾기](/assets/images/Laon/week1-1-2.png){: width="70%" height="auto" .image-center}<br><br>
~~정답은 5개입니다~~
<br><br>
왜 갑자기 틀린그림찾기를 했을까요?
<br><br><br>
우리는 학습을 할때 중요한 특징을 잡습니다. 즉, 어떠한 특징들에게 가중치를 두고 학습을 하는데 우리는 저 그림에서 '틀린그림찾기'의 목적에 맞춰서 <u>두 그림의 다름</u>에 가중치를 두고 판단을 내립니다. 그 결과로 정답을 도출해내는 것이죠. 우리는 이러한 메커니즘을 기계에게도 적용시키고자 합니다.
<br><br><br>

#### Hebbian Rule
<br>
- 학습이란 시냅스 연결의 세기를 결정하는 것입니다.
<br><br>
- 2개의 연결된 세포가 동시에 활성화 되는 경우네는 시냅스의 세기를 올리는것입니다.
<br><br>

이것이 바로 Hebbian Rule 입니다.
<br><br>
즉, 학습의 중요한 요소인 시냅스(가중치(weight))와 세포체(활성함수(activation funtion))의 개념을 갖는 인공뉴런을 만든다면, 학습이 가능해지는 것니다!
<br><br>
## 3. 인공 신경망
<br>
#### 생물학적 세포와 인공세포
<br>
![생물학적 세포와 인공 세포](/assets/images/Laon/week1-1-10.png){: width="70%" height="auto" .image-center}<br><br>
#### 인공 신경망 구조, 퍼셉트론(Perceptron)
<br>
![인공신경망](/assets/images/Laon/week1-1-3.png){: width="70%" height="auto" .image-center}
<br><br>
- 특정 threshold 이상이면 반응하고, 그 이하는 무시되는 기능을 활성함수(activation function)이라고 합니다.
<br><br>
- 선형적인 성질 > 선형적으로 구별이 가능(linearly separable) > 분류에서 유용
<br><br>
- 하지만 이는 XOR문제를 풀 수 없습니다.
<br><br>

![and,or,not and xor](/assets/images/Laon/week1-1-4.png){: width="70%" height="auto" .image-center}
<br><br>
그림에서 처럼 XOR 문제는 선형적으로 구분하기 어렵습니다.
<br><br>
_(하지만 single layer가 아니라 multi-layer를 이용하면 구현할 수 있습니다)_
<br><br>
## 4. 활성 함수 (Activation Function)
<br>
활성 함수에는 다양한 종류가 있습니다.
<br><br>
![활성함수](/assets/images/Laon/week1-1-5.png){: width="70%" height="auto" .image-center}
<br><br>

'Sign Fuction'이나 'Step Function'은 단지 두개의 값만 가능하기 때문에 표현의 한계가 있습니다.
<br><br>
'Linear Function'을 사용할지라도 비선형 조건을 표현할 수 없습니다.
<br><br>
여러 장점들 때문에 주로 'Sigmoid Function'이나 'Hyperbolic Tangent Function'과 같은 비선형 함수를 활성함수로 사용합니다.
<br><br>
하지만 'Sigmoid Function'을 사용한다 해도 망이 깁어지는 Deep Neural Network에서는 학습의 어려움으로 인해 ReLU(Rectifier Linear Unit)을 주로 사용합니다.
<br><br>
#### Rectifier Linear Unit (ReLU)
<br>
![ReLU](/assets/images/Laon/week1-1-6.png){: width="70%" height="auto" .image-center}
<br><br>

## 5. 머신 러닝의 framework

#### 머신 러닝과 일반적인 프로그래밍의 차이
<br>
![머신러닝과의 차이](/assets/images/Laon/week1-1-7.png){: width="70%" height="auto" .image-center}
<br><br>

일반적인 프로그래밍은 데이터와 프로그램을 주면 함수를 도출해내지만, 머신러닝은 데이터와 결과값을 주면 프로그램을 도출합니다.
<br><br>
#### 머신 러닝 framework
<br>
- 머신 러닝이란 학습 집합(training set)을 이용해, 예측함수 'f'를 만들어 내는 과정이라고 생각할 수 있습니다.
<br><br>
- 학습 데이터에 결과값를 달아주는 것을 'label을 달아준다'라고 표현하며, label이 달려있는 데이터를 이용한 학습법을 지도학습(Supervised Learning)이라고 합니다.
<br><br>
- 학습의 질을 높이려면, 학습 데이터가 많아야 합니다.
<br><br>

#### 머신 러닝의 과정
<br>
- 학습(Training)
<br><br>
![Training](/assets/images/Laon/week1-1-8.png){: width="70%" height="auto" .image-center}
<br><br>

- 검사(Testing)
<br><br>
![Testing](/assets/images/Laon/week1-1-9.png){: width="70%" height="auto" .image-center}
<br><br>

- 검증(Validation)
<br><br>
검증단계에선 학습과 다른 데이터를 이용해 학습이 제대로 되었는지를 확인합니다.
<br><br>
_(너무 주어진 학습데이터에만 특화 데이타면, 학습 데이터에서 조금만 입력이 들어오더라도 결과가 나쁘게 나올 수 있습니다. 이를 '오버피팅(Overfitting)'이라고 합니다.)_
<br><br>

#### 학습은 일반화(Generalization)의 과정
<br>
- 일반화를 잘 시키려면 양잘의 많은 학습 데이터와 좋은 알고리즘이 필요합니다.
<br><br>
- 학습에 필요한 많은 시스템 변수(Hyper-parameter)를 설계자가 잘 설정해줘야 합니다.
<br><br>

<b>설계자의 역할</b>은 <u>좋은 알고리즘</u>을 선별하여 사용하고 <u>질 좋고 많은 학습 데이터</u>를 모델에 주는 것입니다.
