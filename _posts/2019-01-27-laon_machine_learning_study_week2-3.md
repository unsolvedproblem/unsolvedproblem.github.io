---
layout: post
title:  "[쉽게읽는 머신러닝-라온피플] 5. Support Vector Machine"
date:   2019-01-27
category: laon
tags: laon
author: Polar B, 백승열
comments: true
---
<br><br>
지난 포스트에는 머신러닝 중 한가지인 Naive Bayes에 대해서 알아보았습니다.
<br>
[[쉽게읽는 머신러닝-라온피플] 4. Naive Bayes](/laon/2019/01/27/laon_machine_learning_study_week2-2.html)
<br>
오늘은 또다른 머신러닝의 강력한 모델 중 하나인 Support Vector Machine(SVM)에 대해 알아보는 시간을 가지겠습니다!
<br><br><br>
가즈아ㅏㅏㅏ!
<br><br><br>

## 1. Support Vector Machine
<br>

#### 역사
<br>
Vladimir N. Vapnik과 Alexey Ya. Chervonekis에 의해 개발되었습니다.
<br><br>
인공 신경망에 비해 간결하고 뛰어난 성능을 보여서 90년대 들어 각광을 받았습니다.

<br><br>

#### 소개
<br>
Support Vector와 Hyper-plane이 주요 개념인 Machine Learning Algorithm 중 하나이며, 지도 학습 모델입니다.
<br><br>

#### 특징
<br>
분류(classification)나 회귀분석(regression)에 사용이 가능합니다.
<br>
Hyper-plane을 이용해 카테고리를 나눕니다.
<br><br>

#### 역할
<br>
주어진 데이터 집합을 바탕으로 새로운 데이터가 어느 집합에 속할지 판단하는 비확률적 이진 선형 분류모델을 만듭니다.
<br><br>
데이터가 사상된 공간에서 분류모델은 경계로 표현이 되는데 SVM알고리즘은 그 중 가장 폭(Margin)이 큰 경계를 찾는 알고리즘입니다.
<br><br>

#### Linearly Separability
<br>

![Linearly Separability](/assets/images/Laon/week2-3-1.png){: width="70%" height="auto" .image-center}
<br><br>

위 그림에서 알 수 있듯이, 데이터를 선형적으로 구분이 가능한 것을 Linearly separable 하다고 합니다.
<br><br>

## 2. Support Vector와 Hyper-plane
<br>

아래 그림에서 어느것이 가장 분류가 잘 되어 보이나요?
<br><br>

![Support Vector & Hyper-plane1](/assets/images/Laon/week2-3-2.png){: width="70%" height="auto" .image-center}
<br><br>

네! 오른쪽 아래 그림입니다. 경계선을 기준으로 데이터 까지의 거리가 가장 멀리 적절하게 떨어져 있기 때문이죠.
<br><br>
SVM은 이런 최대의 Margin을 가진 경계를 구합니다. 그렇게 해야 새로운 데이터가 들어와도 잘 분류할 가능성이 커지기 때문이죠.
<br><br>

![Support Vector & Hyper-plane2](/assets/images/Laon/week2-3-3.png){: width="70%" height="auto" .image-center}
<br>

- Hyper-plane :
<br>
주어진 공간(Ambient Space)에서 그보다 하나 작은 차원의 종속공간(Subspace)을 뜻합니다. 즉, 3차원 공간의 Hyper-plane은 2차원의 평면이 되는 것이고, 2차원 평면의 Hyper-plane은 1차원인 선이 되는 것입니다.
<br><br>
- Support Vector :
<br>
Hyper-plane으로부터 가장 가까이 있는 데이터
<br><br>
- Margin (in SVM) :
<br>
Support Vector와 Hyper-plane사이의 거리
<br><br>

마진의 폭은 $2/(‖𝑤‖)$ 입니다.
<br>
즉, 마진의 폭이 최대가 되게 하려면 ‖𝑤‖가 최소가 되어야 합니다.
<br>
‖𝑤‖ 최소 -> ‖𝑤‖<sup>2</sup> 최소-> quadratic optimization(최적화 방법)
<br><br>




<br>
<br>
<br>
<br>

출처 :
<br>
[쉽게 읽는 머신 러닝 - 라온피플](https://laonple.blog.me/220867768192)
