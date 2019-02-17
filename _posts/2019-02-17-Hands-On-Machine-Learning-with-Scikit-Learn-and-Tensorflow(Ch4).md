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

안녕하세요. 팀 언플(Team Unsolved Problem)의 또다른 에디터인 Polar b 입니다! 오늘 포스팅할 것은 헨즈온 머신러닝의 4장, Training Models, 입니다. 팀 일원인 Duck군의 글을 정리해서 다시 업데이트하는 글이 되겠습니다!
<br><br>

오늘은 지난 시간 'Chapter 3. 분류'에 이어서 'Chapter 4. 모델 훈련'에 들어가겠습니다.
<br>
[[Hands-on ML] Chapter 2. End-to-End Machine Learning Project6](https://unsolvedproblem.github.io/hands-on%20ml/2019/02/12/Hands-On-Machine-Learning-with-Scikit-Learn-and-Tensorflow.html)
<br><br>

그럼, 출발해볼까요?!
<br><br>

4.0 Introduction
<br>
4.1 선형 회귀(Linear Regression)
<br>
  - 4.1.1 정규 방정식
  - 4.1.2 계산 복잡도
4.2 경사 하강법
<br>
  - 4.2.1 배치 경사 하강법
  - 4.2.2 확률적 경사 하강법
  - 4.2.3 미니배치 경사 하강법
4.3 다항 회귀
<br>
4.4 학습 곡선
<br>
4.5 규제가 있는 선형 모델
<br>
  - 4.5.1 릿지 회귀
  - 4.5.2 리쏘 회귀
  - 4.5.3 엘라스틱넷
  - 4.5.4 조기 종료
4.6 로지스틱 회귀
<br>
  - 4.6.1 확률 추정
  - 4.6.2 훈련과 비용 함수
  - 4.6.3 결정 경계
  - 4.6.4 소프트 맥스 회귀
<br><br>

이번 포스트에서는 4.4 학습 곡선까지만 다룰 예정입니다.

##4.0 Introduction
<br>

앞장과는 달리 이제는 실제 모델과 훈련 알고리즘이 어떻게 작동하는지 살펴 볼 것입니다.

## 4.1 선형회귀
<br>

선형 회귀(Linear Regression)는 종속변수 y와 한 개 이상의 독립 변수(또는 설명 변수) X와의 선형 상관 관계를 모델링하는 회귀분석 기법입니다.
<br><br>
