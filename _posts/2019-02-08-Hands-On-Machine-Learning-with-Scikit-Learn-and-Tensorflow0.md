---
layout: post
title:  "[Hands-On ML] Chapter 2. End-to-End Machine Learning Project3"
date:   2019-02-08
category: hands-on ML
tags: hands-on ML
author: Khel Kim, 김현호
comments: true
---
<br><br>
[코딩관련깃허브](https://github.com/rickiepark/handson-ml)
<br><br>
안녕하세요. 팀 언플(Team Unsolved Problem)에 에디터 ㅋ헬 킴(Khel Kim), 김현호입니다.
<br>
오늘은 지난 시간 Section 2.3 Take a Quick Look at the Data Structure에 이어서 Section 2.4 Create Test Set에 들어가겠습니다.
<br>
[[Hands-on ML] Chapter 2. End-to-End Machine Learning Project2](https://unsolvedproblem.github.io/hands-on%20ml/2019/02/08/Hands-On-Machine-Learning-with-Scikit-Learn-and-Tensorflow.html)
<br><br><br>
그럼, 시작하겠습니다!
<br><br><br>
1. Look at the big picture.
<br><br>
2. __Get the data.__
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



## 2.4 Create a Test Set
<br>
이 섹션에서는 데이터 셋을 훈련 세트(train set)와 테스트 세트(test set)로 나눌 것입니다. 라온 피플 포스팅에선 데이터는 다다익선처럼 말해놓고 굳이 훈련 세트와 테스트 세트로 나눈다니... 왜 굳이 이런 짓을 하는 지 이해가 잘 되지 않을 수도 있습니다.<br><br>
하지만 이 단계는 매우 중요한 단계입니다. 왜냐하면 우리의 데이터를 보고 편견이 생겨 적합하지 못한 알고리즘을 쓸 수도 있기 때문입니다. 또, 우리가 만약 가지고 있는 모든 데이터를 알고리즘에 넣게 된다면 우리는 알고리즘의 성능을 측정할 수가 없게 됩니다(왜냐하면 우리가 가진 데이터 내에서는 모두 좋은 결과값을 낼 테니까요). 따라서 우리의 알고리즘이 정말 '예측' 능력이 좋은지 확인하기 위해선 테스트 세트(알고리즘이 보지 못한)를 가지고 있어야 합니다. 그리고 우리의 프로젝트 마지막 단계에서 이 테스트 세트로 우리의 알고리즘을 평가해야겠죠.<br><br>
이번 포스팅에서는 우리의 데이터를 훈련 세트와 테스트 세트로 나누는 여러가지 방법을 알아 보겠습니다.
<br>
사실 이 테스트 세트를 생성하는 일은 어렵지 않습니다. 그저 가지고 있는 데이터 중에 20% 정도 랜덤으로 샘플을 뽑아내면 됩니다. 코드를 보시죠.<br><br>  
~~~
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
~~~
<br>
코드를 읽어보겠습니다.<br>

~~~
numpy 모듈을 불러옵니다(행렬 혹은 행렬 연산을 위해 필요한 모듈입니다).

split_train_test 함수를 정의합니다.
  shuffled_indices 변수에 data의 개수만큼의 수를 랜덤하게 섞어 저장합니다.
  test_set_size 변수에 전체 데이터에 원하는 테스트 세트의 비율을 곱한 값의
  소수점을 떼고 저장합니다.
  test_indices 변수에 shuffled_indices의 앞 부분(테스트 세트 사이즈만큼)을
  떼어 저장합니다.
  train_indices 변수에 shuffled_indices의 나머지 부분을 저장합니다.
  함수값으로 데이터 중 train_indices의 인덱스를 가진 데이터와
  test_indices의 인덱스를 가진 데이터를 튜플로 묶어 뱉어냅니다.
~~~
<br>
이 함수는 다음과 같이 사용할 수 있습니다.<br>
~~~
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
~~~
<br><br>
하지만 이 함수의 문제는 매번 이 함수를 돌릴 때마다 다른 테스트 세트를 만드는 것입니다. 이 방법을 해결하는 두가지 방법이 있습니다.
1. 테스트 세트를 저장합니다.
2. random seed를 설정할 수 있습니다.


<br>
하지만 이 두 방법 또한 문제가 있습니다. 데이터가 업데이트되면 똑같은 문제가 발생합니다. 일반적인 해결 방법은 각 객체들의 변경 불가능한 값들을 기준으로 테스트 세트로 보낼지 말지 결정하는 것입니다. 예를 들어 각 샘플마다 인덱스의 해시값을 계산하여 (샘플이 변경 불가능한 인덱스를 가졌다고 가정합시다) 해시의 마지막 바이트의 값이 51(256의 20% 정도)보다 작거나 같은 샘플만 테스트 세트로 보낼 수 있습니다.<br><br>
~~~
import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
~~~
<br>
코드를 읽어보겠습니다.<br>
~~~
hashlib 모듈을 불러옵니다(md5나 sha1으로 hash 변환 해주거나, hash를 디코딩 해줍니다).

test_set_check 함수를 정의합니다(hash는 밑에 함수에서 따로 정의해줌).
  identifier의 원소들을 정수화하고 hash화 한 다음 digest()로 바이트로 만들고
  바이트 마지막 부분을 숫자로 나타낸 다음, 256 * test_ratio와 비교 후 bool값을
  뱉습니다.

split_train_test_by_id 함수를 정의합니다.
  ids 변수에 data에 id_column을 저장합니다.
  in_test_set 변수에 ids의 각각의 원소에 test_set_check 함수를 적용하여 저장합니다.
  data에서 in_test_set에 값이 False인 위치에 있는 객체들과
  data에서 in_test_set에 값이 True인 위치에 있는 객체들을 튜플로 반환합니다.

housing 데이터프레임에 index 컬럼을 추가합니다.

(train_set, test_set)에 split_train_test_by_id의 결과값을 저장합니다.
~~~
<br><br>
우리가 만약 인덱스를 식별자로 사용했다면, 기존 데이터의 마지막에 새로운 데이터가 위치해야하고 어떤 데이터도 삭제되지 않아야 합니다. 만약 이게 불가능하다면 다른 특성을 식별자 컬럼으로 사용해야 합니다. 그리고 그 컬럼은 타겟값과 상관관계가 없어야만 합니다(예를 들어 위도나 경도에 경우 집 값에 영향을 줄 수도 있으므로 적절하지 않습니다).
<br><br>
sklearn을 통해서 훈련 세트와 테스트 세트를 나눌 수 있습니다.

~~~
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
~~~
<br>
여기서 random_state는 random seed입니다.
<br><br>
지금까지의 임의 추출 방법은 데이터가 크면 문제가 없지만 만약 아니라면 좋은 방법이 아닐 수 도 있습니다.
<br><br>
왜냐하면 미국 인구의 60%가 여성이고 40%가 남자라면 샘플을 뽑을 때도 여성 60%, 남성 40% 비율을 유지하는 것이 좋기 때문입니다. 이를 <b>계층적 샘플링</b>이라고 합니다.
<br><br>
만약 우리가 median_income이 집 값을 예측하기에 좋은 특성이라는 것을 알게 되었다면, 이를 기준으로 계층적 샘플링을 하는 것이 좋습니다.
- 이 경우에는 median_income이 연속형 변수이므로, 먼저 이를 명목형 변수로 바꿔주는 것이 좋습니다.

~~~
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
~~~
<br>
코드를 읽어보겠습니다.<br>
~~~
housing 데이터에 income_cat이라는 특성을 새로 만들고 그 특성에 median_income을 1.5로 나누고 소수 첫 번째 자리에서 올림한 수들을 저장합니다(명목형 변수화).
housing 데이터에 income_cat 원소들 중 where 안쪽 조건문이 True인 위치(housing["income_cat"] < 5)에 있는 원소는 그대로 두고 False인 위치에 있는 원소는 주어진 값(5.0)으로 대체합니다.
## pd.DataFrame.where(bool값을 가진 시리즈, 값, inplace=True)
~~~
<br><br>
이제 데이터를 housing["income_cat"]의 비율을 기준으로 훈련 세트와 테스트 세트로 나눠보겠습니다.
<br>
~~~
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
~~~
<br>
다른 건 별로 문제되지 않지만 n_splits가 어떤 것을 나타내는 지 의아하실 수 있습니다. split.split 함수는 generator를 만듭니다. 따라서 for문이 돌 때마다 train_index와 test_index에 인덱스를 저장하고 strat_train_set과 strat_test_set을 뽑습니다. n_splits은 이러한 인텍스를 몇 번 뽑을 지를 결정합니다.
- 따라서 지금은 for 반복문이 큰 의미(?)를 갖지 않습니다.
<br><br>



마지막으로 우리가 일부로 넣어준 컬럼을 제거합시다.
~~~
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
~~~
<br>
'set'의 income_cat 이름을 가진 컬럼을 지우고(axis=1) 원래 'set'을 대체하는 코드입니다.
<br><br>
정말 마지막으로 우리의 훈련 세트를 housing 변수에 넣고 맨 위 5 객체의 내용을 확인합시다.
<br>
![Top five rows after spliting](/assets/images/Hands-on/ch2fig8.png){: width="100%" height="auto" .image-center}

<br><br>
지금까지 주어진 데이터를 훈련 세트와 테스트 세트로 나누는 여러가지 방법을 배웠습니다. 다음 포스팅에는 주어진 데이터를 시각화하는 작업을 살펴보겠습니다.
<br><br>
