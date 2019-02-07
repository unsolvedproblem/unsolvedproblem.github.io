---
layout: post
title:  "[Hands-On ML] Chapter 2. End-to-End Machine Learning Project1"
date:   2019-02-07
category: hands-on ML
tags: hands-on ML
author: Khel Kim, 김현호
comments: true
---

<br><br>
안녕하세요. 팀 언플(Team Unsolved Problem)에 에디터 ㅋ헬 킴(Khel Kim), 김현호입니다. 라온 피플 포스팅을 백곰 백승열 에디터가 잘 정리해서 올려주었습니다.

오늘부터는 머신러닝의 교과서로 인정받고 있는 '핸즈온 머신러닝'을 함께 공부하고 차례대로 정리해보도록 하겠습니다.

Chapter 1같은 경우에는 저희가 라온 피플 포스팅에서 어느정도 다뤘다고 생각이 되어 생략하고 Chapter 2부터 공부하도록 하겠습니다. Chapter 2는 한 프로젝트를 처음부터 끝까지 작업하는 것이기 때문에 자세히 여러차례 나눠서 진행하도록 하겠습니다.


혹시 Chapter 1 관련해서 질문이 있으시다면 편하게 메일 주시면 됩니다.
<br><br><br>
그럼, 시작하겠습니다!
<br><br><br>

이 단원에서는 우리가 직접 데이터사이언티스트가 되어 한 프로젝트를 처음부터 끝까지 다뤄보도록 하겠습니다. 우리가 다룰 분야의 데이터는 부동산 데이터입니다!

<br><br>
먼저 데이터사이언티스트로서 어떤 스탭들을 밟아나갈지를 살펴보죠.
<br><br>
1. Look at the big picture.
<br><br>
2. Get the data.
<br><br>
3. Discover and visualize the data to gain insights.
<br><br>
4. Prepare the data for Machine Learning algorithms.
<br><br>
5. Select a model and train it.
<br><br>
6. Fine-tune our model.
<br><br>
7. Present our solution.
<br><br>
8. Launch, monitor, and maintain our system.
<br><br>


## 0.0. 실제 데이터를 얻을 수 있는 곳
<br>
우리가 데이터를 얻을 수 있는 소스들입니다.
<br><br>
- Popular open data repositories:
<br><br>
  - UC Irivne Machine Learning Repository
<br><br>
  - Kaggle datasets
<br><br>
  - Amazon's AWS datasets
<br><br>
- Meta portals(they list open repositories):
<br><br>
  - http://dataportals.org/
<br><br>
  - http://opendatamonitor.eu/
<br><br>
  - http://quadl.com/
<br><br>
- Other pages listing many popular open data repositories:
<br><br>
  - Wikipedia’s list of Machine Learning datasets
<br><br>
  - Quora.com question
<br><br>
  - Datasets subreddit
<br><br>


이 단원에서는 California Housing Prices dataset from the Statlib repository를 다루겠습니다!
<br><br>
![California Housing Prices dataset](/assets/images/Hands-on/ch2fig1.png)
