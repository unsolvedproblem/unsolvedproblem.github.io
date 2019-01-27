---
layout: post
title:  "[쉽게읽는 머신러닝-라온피플] 3. Decision Tree"
date:   2019-01-27
category: laon
tags: laon
author: Polar B, 백승열
comments: true
---
<br><br>
지난 포스트에는 머신러닝의 학습 방법과 Bagging, 그리고 Boosting에 대해서 알아봤습니다.
<br>
[[쉽게읽는 머신러닝-라온피플] 2. 머신러닝의 학습방법](/laon/2019/01/23/laon_machine_learning_study_week1-2.html)
<br>
오늘은 머신러닝의 모델 중 하나인 결정트리에 대해 알아보는 시간을 가지겠습니다!
<br><br><br>
그럼, 가 봅시다!!
<br><br><br>

## 1. Decision Tree
<br>

#### 역사
<br>
시드니 대학의 J.Ross QuinLan이 한 모델을 만들었고 그것을 그의 책 'Machine Learning, Vol.1, No.1, in 1975'에 실었습니다.
<br><br>
그의 첫 Decision Tree 알고리즘은 Iterative Dichotomiser3 (ID3)이라고 부릅니다.
<br><br>

#### 소개
<br>
Decision Tree는 나무 모양의 그래플르 사용하여 최적의 결정을 할 수 있도록 돕는 방법(알고리즘)입니다.
<br>
이 알고리즘을 Machine Learning에 적용한 것을 <U>Decision Tree Learning</U> 혹은 그냥 <U>Decision Tree</U>라고 부릅니다.
<br><br>

#### 특징
<br>
기회비용에 대한 고려, 기대 이익 계산, 자원의 효율적 사용이나 위험관리 등 효율적 결정이 필요한 분야에 사용합니다.
<br><br>

#### 역할
<br>
어떤 항목에 대한 관측값(Observation)에 대하여 가지(Branch) 끝에 위치하는 기대 값(Target)과 연결시켜주는 예측 모델(Predictive Model)입니다.
<br><br>

## 2. Elements of Decision Tree
<br>

#### 3대 요소
<br>

![결정트리 예제1](/assets/images/Laon/week2-1-1.png){: width="70%" height="auto" .image-center}
<br><br>

  - 노드(Node) :
<br>
각 노드(node)는 1개의 속성(attribute) X<sub>i</sub>를 판별하는 역할
<br><br>
  - 가지(Branch) :
<br>
각 가지(branch)는 각 노드로부터 나오며, 속성(attribute) X<sub>i</sub>에 대한 1개의 값을 가짐
<br><br>
  - 잎(Leaf Node) :
<br>
잎(leaf node) 는 최종단. 입력 데이터가 X<sub>i</sub> 일때 그것에 대응하는 기대값 y에 해당
<br><br>

(예시에서) 습도, 날씨, 바람은 판단의 기준입니다.
<br>
목적에 따라 많은것이 올 수 있습니다.
<br>
속성엔 여러 값이 있을 수 있습니다.
<br><br>

Decision Tree의 다른 예제 :
<br>
![결정트리 예제2](/assets/images/Laon/week2-1-2.png){: width="70%" height="auto" .image-center}
<br><br>

## 3. Decision Tree 만들기와 엔트로피
<br>

Decision Tree를 구성하는 방법은 여러가지 입니다. 속성이 여러 개 있는 경우 <U>어떤 속성을 Root Node(최상단 노드)에 둘지 중요</U>합니다. 일반적으로 더 Compact하게 만드는 것이 목적이고 이를 위해 '<U>엔트로피</U>'를 이용합니다.
<br><br>

![결정트리 예제2](/assets/images/Laon/week2-1-3.png){: width="30%" height="auto" .image-center}
<br>
(P<sub>i</sub> : 특정값 i가 일어날 확률)
<br><br>

## 4. Decision Tree만들기 예제
<br>

![결정트리 만들기 예제1](/assets/images/Laon/week2-1-4.png){: width="70%" height="auto" .image-center}
<br><br>

위의 표는 14일 동안 골프를 치거나 치지 않을 경우에 조건들을 담고 있습니다. Decision Tree를 이용하면 깔끔하게 정리할 수 있고 쉽게 예측할 수 있습니다.
<br><br>

먼저 우리가 판단해야 할 것은 골프를 칠것인가 안칠것인가 하는 것입니다. 예제의 4개의 속성은 Temperature, Outlook, Humidity, Windy입니다. 최적의 Decision Tree를 만들려면 각각의 속성에 대한 엔트로피를 계산해야 합니다.
<br><br>

계산 결과,
- H<sub>temperature</sub> : 1.07
- H<sub>outlook</sub> : 1.09
- H<sub>humidity</sub> : 0.69
- H<sub>windy</sub> : 0.68
<br>

가 나옵니다.
<br><br>

즉, Outlook이 최고 높은 엔트로피를 가지므로 Outlook을 Root Node로 결정합니다. 그 후 Outlook 속성에 있는 3가지 값을 사용하여 같은 방식으로 다음 노드를 결정합니다.
<br><br>

결과적으로...
<br><br>
![결정트리 만들기 예제2](/assets/images/Laon/week2-1-5.png){: width="70%" height="auto" .image-center}
<br><br>
이런 모양의 Decision Tree가 완성이 됩니다!
<br><br>

#### 엔트로피
<br>

![엔트로피](/assets/images/Laon/week2-1-6.png){: width="70%" height="auto" .image-center}
<br><br>

위 그림은 속성이 2개인 경우에 데이터의 분포에 따른 엔트로피의 변화를 보여주는 그림입니다.
<br><br>
분포가 고르면 큰 값을 가지고, 특정값으로 몰려 있으면 0에 가까워 집니다.
<br><br>
코딩에 필요한 bit효율(log<sub>2</sub>를 쓰는 Entorpy의 단위가 bit)이 올라간다는 것은 엔트로피가 올라간다는 것이고, 결과적으로 효율적인 Decision Tree가 만들어진다는 의미입니다.
<br><br>

## 5. Decision Tree의 Overfitting
<br>

![Overfitting1](/assets/images/Laon/week2-1-7.png){: width="70%" height="auto" .image-center}
<br><br>
통상적으로 크기(노드의 개수)가 대략적으로 23 이상이 되면 Test에 대한 정확도가 점점 감소합니다.
<br><br>
트리가 커지면 세밀한 분류가 가능하지만, Overfitting될 가능성이 높아집니다.
<br><br>

#### Overfitting의 예
<br>

![Overfitting2](/assets/images/Laon/week2-1-8.png){: width="70%" height="auto" .image-center}
<br><br>
이 표를 통해서 포유류를 구분하는 모델을 만들생각입니다. 어랏! 그런데 표에 데이터 잡음으로 인해 Bat과 Whale에 엉뚱한 Label이 붙었습니다. 이것을 이용해 Decision Tree를 만들면...
<br><br>

![Overfitting3](/assets/images/Laon/week2-1-9.png){: width="70%" height="auto" .image-center}
<br><br>

이 모델을 통해서 다음 주어진 테스트 데이터로 테스트를 하면...
<br><br>

![Overfitting4](/assets/images/Laon/week2-1-10.png){: width="70%" height="auto" .image-center}
<br><br>

사람과 돌고래는 다리가 4개가 아니기 때문에 포유류가 아니라고 하는군요!
<br><br>

#### Overfittin 피하는 방법 - 가지치기(Pruning)
<br>

가지치기를 실행하면 학습데이터에 대한 Error가 발생합니다. 하지만 테스트 데이터에 대한 Error는 감소하죠.
<br><br>
학습 데이터를 통해 학습시킨 후 결과 검증을 위한 검증 데이터를 이용해 학습 결과의 특화 여부를 판단하고, 선 가지치기(Pre-Pruning)와 (후 가지치기)Post-Pruning을 시행합니다.
<br><br>

  - Pre-Pruning :
<br>
몇몇 측정결과 후 이를 바탕으로 Sub Tree의 구축을 정지합니다. 이 측정 결과에는 Information gain, Gini index 등이 있습니다.
<br><br>
  - Post-Pruning :
<br>
Decision Tree를 완전히 만듭니다. 그 후에 바닥부터 위로 손질을 합니다. 노드를 잎으로 대체하는 식의 가지치기로 진행합니다.
<br><br>

가지치기를 한 후의 Model :
<br><br>

![Overfitting5](/assets/images/Laon/week2-1-11.png){: width="70%" height="auto" .image-center}
<br><br>

가지치기를 했을 경우의 모델 정확도 그래프 :
<br><br>

![Overfitting5](/assets/images/Laon/week2-1-12.png){: width="70%" height="auto" .image-center}
<br><br>

## 6. Decision Tree의 장단점
<br>

  - 장점
    - 빠르게 구현가능
    - 특징의 유형에 상관없이 잘 동작
    - 특이점(Outlier)에 대하여 상대적으로 덜 민감
    - 튜닝이 필요한 파라미터의 개수가 작음
    - 값이 빠진 경우도 효율적으로 처리가 가능
    - 해석이 쉬워짐
<br><br>

  - 단점
    - Only axis-alinged splits of data
    - Greedy(may not find best tree. Exponentially many possible trees.)


참고 영상 :
<br><br>

<iframe width="80%" height="400" src="https://www.youtube.com/embed/A-iqpbz7IDE?list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>{: .image-center}

<br>
<br>
<br>
<br>

출처 : [쉽게 읽는 머신 러닝 - 라온피플](https://laonple.blog.me/220861527086)
