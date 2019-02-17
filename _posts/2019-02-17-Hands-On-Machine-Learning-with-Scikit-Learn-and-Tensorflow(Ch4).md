---
layout: post
title:  "[Hands-On ML] Chapter 4. Training Models"
date:   2019-02-17
category: hands-on ML
tags: hands-on ML
author: Polar b, 백승열
comments: true
---
<br><br>
[코딩관련깃허브](https://github.com/rickiepark/handson-ml)
<br><br>
_Author : Duck Hyeun, Ryu_
<br>
안녕하세요. 팀 언플(Team Unsolved Problem)의 또다른 에디터인 Polar b 입니다! 오늘 포스팅할 것은 헨즈온 머신러닝의 4장, Training Models, 입니다. 팀 일원인 Duck군의 글을 정리해서 다시 업데이트하는 글이 되겠습니다!
<br><br>

오늘은 지난 시간 'Chapter 3. 분류'에 이어서 'Chapter 4. 모델 훈련'에 들어가겠습니다.
<br>
[[Hands-on ML] Chapter 2. End-to-End Machine Learning Project6](https://unsolvedproblem.github.io/hands-on%20ml/2019/02/12/Hands-On-Machine-Learning-with-Scikit-Learn-and-Tensorflow.html)
<br><br>

그럼, 출발해볼까요?!
<br><br>

- 4.0 Introduction
- 4.1 선형 회귀(Linear Regression)
  - 4.1.1 정규 방정식
  - 4.1.2 계산 복잡도
- 4.2 경사 하강법
  - 4.2.1 배치 경사 하강법
  - 4.2.2 확률적 경사 하강법
  - 4.2.3 미니배치 경사 하강법
- 4.3 다항 회귀
- 4.4 학습 곡선
- 4.5 규제가 있는 선형 모델
  - 4.5.1 릿지 회귀
  - 4.5.2 리쏘 회귀
  - 4.5.3 엘라스틱넷
  - 4.5.4 조기 종료
- 4.6 로지스틱 회귀
  - 4.6.1 확률 추정
  - 4.6.2 훈련과 비용 함수
  - 4.6.3 결정 경계
  - 4.6.4 소프트 맥스 회귀
<br><br>

이번 포스트에서는 4.4 학습 곡선까지만 다룰 예정입니다.

## 4.0 Introduction
<br>

앞장과는 달리 이제는 실제 모델과 훈련 알고리즘이 어떻게 작동하는지 살펴 볼 것입니다. 가장 간단한 모델중 하나인 선형 회기부터 시작해서 다항 회귀를 살펴보고 모델 학습에서 발생할 수 있는 과대적합 문제를 해결할 수 있는 규제 기법을 알아보겠습니다. 끝으로 분류에 널리 쓰이는 로지스틱 회기와 소프트맥스 회귀를 알아보겠습니다.
<br><br>

## 4.1 선형회귀
<br>

선형 회귀(Linear Regression)는 종속변수 y와 한 개 이상의 독립 변수(또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀분석 기법입니다.
<br><br>

**선형 모델의 예측**
<br>

<center>$$ \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n $$</center>
<br>
- $\hat{y}$ : 예측값
- $n$ : 특성의 수
- $x_i$ : $i$번째 특성값
- $\theta_j$ : $j$번째 모델 파라미터
<br><br>

선형 모델의 예측 벡터형태
<br>
<center>$$ \hat{y} = h_{\theta}(x) = \theta^T \dot x $$</center>
<br>
- $\theta$ : 편향 $\theta_0$와 $\theta_1$에서 $\theta_n$까지의 특성 가중치를 담고 있는 모델의 파라미터
- $\theta^T$ : $\theta$의 전치(Transpose)
- $x$ : $x_0$에서 $x_n$까지 담고있는 샘플의 특성 벡터($x_0$는 항상 1)
- $h_\theta$ : 모델 파라미터 $\theta$를 사용한 가설 함수
<br>
*(편의상 벡터 표현식 $x$의 성분 중 첫번째 $x_0$는 1이라 생각합니다.)*
<br><br>

위의 식이 바로 선형 회귀 모델입니다. 이제 훈련을 시켜야겠죠? 모델을 훈련시킨다는 뜻은 모델이 훈련세트에 가장 잘 맞도록 모델 파라미터를 설정하는 것입니다. 그러기 위해선 모델의 예측값이 얼마나 실제 타겟값과 비슷한지(즉, 모델의 성능이 얼마나 좋은지) 알 수 있어야합니다.
<br><br>
그것을 알게 해주는 것이 바로 회귀에서 가장 널리 쓰이는 성능 측정 지표인 평균 제곱근 오차(RMSE) 입니다.(2장을 다루는 포스터중 첫번째 포스트를 확인해보시기 바랍니다.)따라서 선형 회기 모델을 훈련시킨다는 뜻은 RMSE를 최소화하는 $\theta$를 찾아낸다는 것입니다. RMSE와 평균 제곱 오차(Mean square error, MSE)는 최소화하는 것이 같은 결과는 내지만 MSE가 더 간단합니다.

평균 제곱 오차 비용함수(Mean square error cost function)
<br>
<center>$$ MSE(X,h_\theta) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T \dot x_i - y_i)^2 $$</center>
<br>
- $m$ : 선형모델을 훈련시킬 데이터 수
- $\theta^T \dot x_i$ : i번째 데이터의 예측값
- $y_i$ : $i$번째 데이터의 실제 타겟값
<br><br>
