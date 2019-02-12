---
layout: post
title:  "[Hands-On ML] Chapter 2. End-to-End Machine Learning Project6"
date:   2019-02-12
category: hands-on ML
tags: hands-on ML
author: Khel Kim, 김현호
comments: true
---
_Author : Duck Hyeun, Ryu_
<br><br>
안녕하세요. 팀 언플(Team Unsolved Problem)에 에디터 ㅋ헬 킴(Khel Kim), 김현호입니다.
<br>
오늘은 지난 시간 Section 4. Prepare the data for Machine Learning algorithms에 이어서 Section 5. Select a model and train it에 들어가겠습니다.
<br>
[[Hands-on ML] Chapter 2. End-to-End Machine Learning Project4](https://unsolvedproblem.github.io/hands-on%20ml/2019/02/09/Hands-On-Machine-Learning-with-Scikit-Learn-and-Tensorflow1.html)
<br><br><br>
그럼, 시작하겠습니다!
<br><br><br>
1. Look at the big picture.
<br><br>
2. Get the data.
<br><br>
3. Discover and visualize the data to gain insights.
<br><br>
4. Prepare the data for Machine Learning algorithms.
<br><br>
5. __Select a model and train it.__
<br><br>
6. Fine-tune our model.
<br><br>
7. Present our solution.
<br><br>
8. Launch, monitor, and maintain our system.
<br><br>



## 5.0 Select a model and train it
<br>
이제 머신러닝 알고리즘을 위해 데이터를 전처리해보겠습니다. 이 단계는 최대한 자동화해야 하는데 이유를 설명해 드리겠습니다.
1. 어떤 데이터 셋에 대해서도 데이터 변환을 손쉽게 반복할 수 있음
<br><br>
2. 향후 프로젝트에 사용할 수 있는 변환라이브러리의 점진적 구축 가능
<br><br>
3. 실제 시스템에서 알고리즘에 주입하기 전에 데이터를 변환시키는데 이 함수 사용 가능
<br><br>
4. 여러 가지 데이터 변환을 쉽게 시도할 수 있음
<br><br>
5. 어떤 조합이 가장 좋은지 확인하는데 편함  
<br><br>


하지만 먼저 housing을 원래 훈련 세트로 복원하고, 예측 변수와 타깃 값에 같에 같은 변형을 적용하지 않기 위해 예측 변수와 레이블을 분리합시다.
<br><br>
~~~
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
~~~
<br>
코드를 한번 읽어보겠습니다.
<br>
~~~
housing에 median_hous_value(타깃 값)을 뺀 복사본(얕은 복사)을 넣습니다. axis=1은 column방향으로 데이터를 삭제하겠다는 뜻입니다.
housing_labels에 median_income(타깃 값)을 얕은 복사로 넣습니다.
~~~
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
## 5.1 Data Cleaning
<br>
