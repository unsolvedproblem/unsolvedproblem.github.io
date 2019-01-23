---
layout: post
title:  "[쉽게읽는 머신러닝-라온피플] 2. 머신러닝의 학습방법"
date:   2019-01-23 17:16:00 -0000
category: laon
tags: laon
author: Polar B, 백승열
comments: true
---
<br><br>
지난 포스트에는 머신러닝의 framework에 대해서 알아보았습니다.
<br>
[[쉽게읽는 머신러닝-라온피플] 1. 머신러닝 framework](/laon/2019/01/22/laon_machine_learning_study_week1-1.html)
<br>
오늘은 머신러닝의 학습방법과 몇가지 종류의 머신러닝 모델(Boosting and Bagging)에 대해서 알아보겠습니다!
<br><br><br>
그럼, 가 봅시다!!
<br><br><br>

## 1. 머신 러닝의 학습 방법
<br>
먼저, 머신 러닝 알고리즘의 종류는 매우 다양하고 여러 카테고리로 나뉩니다.
<br><br>

![다양한 종류의 ML알고리즘](/assets/images/Laon/week1-2-1.png){: width="70%" height="auto" .image-center}
<br><br>

#### 머신 러닝 알고리즘의 분류
<br>

- 지도 학습(Supervised Learning) :
<br><br>
  - 주어진 입력에 대해 어떤 결과가 나올지 알고 있는 학습 데이터를 이용해, 데이터를 해석할 수 있는 모델을 만들고 그것을 바탕으로 새로운 데이터를 추정하는 방법입니다.
<br><br>
  - 모델 f를 결정하기 위해 많은 변수들을 풀어야 하는데, 변수의 수가 너무 많게 되면 대수(algebra)를 이용해 풀어내는 것이 불가능 합니다. 따라서 수치해석(numerical analysis) 방법으로 최적값을 찾아가는 과정을 반복적으로 수행합니다.
<br><br>

- 자율 학습(Un-supervised Learning) :
<br><br>
  - 학습 데이터에 대한 기대값이 붙어 있지 않기 때문에, 학습 알고리즘을 통해 학습 데이터 속에 있는 어떤 패턴이나 숨어 있는 중요한 핵심 컨셉을 찾아서 학습하는 것입니다.
<br><br>
  - 특정 가이드가 없기 때문에 기대했던 것과 다른 결과가 나올 수도 있습니다.
<br><br>

- 강화 학습(Reinforcement Learning)
<br><br>
  - 훈련을 잘 따르면 '상(reward)'을 주고, 그렇지 못하면 '벌(punishment)'을 주며, 상과 벌을 통해 훈련의 감독관이 원하는 방향으로 학습을 하게 됩니다.
<br><br>
<iframe width="80%" height="400" src="https://www.youtube.com/embed/3bhP7zulFfY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>{: .image-center}
<br><br>

## 2. Boosting
<br>

무작위로 선택하는 것보다 약간 가능성이 높은 규칙(weak learner)들을 결합시켜 보다 정확한 예측 모델을 만들어 내는 것입니다.
<br><br>
_(Weak learner : 무작위로 선정하는 것보다는 성공 확률이 높은, 즉 오차율이 50% 이하인 학습 규칙)_
<br><br>

#### Boosting 학습 방법
<br>

Weak learner를 선정하는 방법은 머신 러닝 알고리즘을 적용하여 서로 다른 분포(distribution)을 갖도록 해주는 것입니다.
<br><br>
매번 기본 러닝 알고리즘을 적용할 때 마다 새로운 Weak learner를 만들며, 이 과정을 반복적으로 수행합니다.
<br><br>
이 Weak learners를 하나로 모아서 Strong learner를 만듭니다.
<br><br>

![Boosting1](/assets/images/Laon/week1-2-2.png){: width="70%" height="auto" .image-center}
<br><br>

Weak learner를 이용해 학습을 하면서 에러가 발생하면, 그 에러에 좀 더 집중하기 위해 error에 대한 weight를 올리고, 그 에러를 잘 처리하는 방향으로 새로운 Weak learner를 학습시킵니다.
<br><br>
최종 결과는
<br><br>
![Boosting2](/assets/images/Laon/week1-2-3.png){: width="70%" height="auto" .image-center}
<br><br>
과 같이 표현되며, 여기서 α<sub>t</sub>는 가중치 입니다.
<br><br>
Boosting은 새로운 Learner를 학습할 때마다 이전 결과를 참조하는 방식이며, 이것이 뒤에 나올 Bagging과 다른 점입니다.
<br><br>
최종적으로 Weak learner로 부터의 출력을 결합하여 더 좋은 예측율을 갖는 Strong learner가 만들어 집니다.
<br><br>

## 3. Bagging (Bootstrap Aggregating)
<br>

#### Bootstrapping
<br>

먼저 표본을 취하고, 그 표본에 대한 분포를 구합니다. 그리고 나서 표본을 전체라고 생각하고, 표본으로 부터 많은 횟수에 걸쳐(동일한 개수의) 샘플을 복원 추출(Resample with replacement)한 후 각 샘플에 대한 분포를 구합니다.
<br><br>
그 후 전체 표본의 분포와 샘플들 간의 분포의 관계를 통해, 전체 집단의 분포를 유추하는 방식입니다.
<br><br>
Regression의 경우는 평균(model averaging)을 취해 분산(variance)를 줄이는 효과를 얻을 수 있고, 분류(classification)에서는 투표 효과(voting)을 통해 가장 많은 결과가 나오는 것을 취하는 방식을 사용합니다.
<br><br>
[Bagging의 예 -위키피디아](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
<br><br>

#### Bagging을 적용하면 안되는 경우
<br>

  - 표본 데이터가 작은 경우 :
  <br><br>
  표본 데이터가 전체 데이터를 잘 반영하지 못합니다.
  <br><br>
  - 데이터에 잡음이 많은 경우 :
  <br><br>
  특이점(outlier)이 추정을 크게 왜곡 시킬 가능성이 있습니다.
  <br><br>
  - 데이터에 의존성이 있는 경우 :
  <br><br>
  기본적으로 Bootstrapping은 데이터가 독립적인 경우를 가정합니다.
  <br><br>

#### Bagging과 Boosting의 차이
<br>

  1. Bagging은 모든 Boostrap이 서로 독립적인 관계를 가집니다. 하지만 Boosting은 순차적으로 처리가 되며, 에러가 발생하면 그 에러의 weight를 올리기 때문에 현재의 Weak learniner가 이전 Weak learner의 영향을 받습니다.
<br><br>
  2. Boosting은 최종적으로 weighted vote을 하지만, Bagging은 단순 vote을 합니다.
<br><br>
  3. Bagging은 분산을 줄이는 것이 주 목적이지만, Boosting은 바이어스를 줄이는 것이 주 목적입니다.
<br><br>
  4. Bagging은 Overfitting문제를 해결 할 수있지만, Boosting은 Overfitting의 문제로 부터 자유롭지 못합니다.
<br><br>
<br><br>

_(Bagging과 Boostng에 대해서는 추후에 Hands-on Machine learning 책을 정리할 때 더 자세히 다루겠 습니다.)_

<br>
<br>
<br>
<br>

출처 : [쉽게 읽는 머신 러닝 - 라온피플](https://laonple.blog.me/220827900759)
