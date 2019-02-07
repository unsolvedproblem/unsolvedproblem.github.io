---
layout: post
title:  "Hackerrank 30 Days of Code 8 [Dictionaries and Maps]"
date:   2019-02-07
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Dictionaries and Maps](https://www.hackerrank.com/challenges/30-dictionaries-and-maps/problem)

~~~
if __name__ == "__main__":
    n = int(input())
    arr = tuple([input().split(' ') for _ in range(n)])
    dic = {}
    for name, number in arr:
        dic[name] = number

    while True:
        try:
            given_name = input()
            if given_name in dic.keys():
                print('%s=%s' %(given_name, dic[given_name]))
            else:
                print("Not found")
        except EOFError:
            break
~~~

이 딕셔너리 자료형을 할 때 필요했던 개념이 있었습니다.  
혹시 다른 분들도 필요하실까 제가 정리해 본 것을 올려보겠습니다.  

Day 8

어떤 데이터를 읽어와 input() 함수에 넣는 형식.

하지만 몇 개의 데이터가 있는지 알 수 없는 경우 특별한 방법을 사용해야한다.

~~~
while True:
    tmp = input()
    if tmp:
        print(tmp)
    else:
        break
~~~
하지만 이러한 코드의 약점은 input을 해주는 사람이 마지막에 빈칸을 넣어줘야 코드를 빠져나올 수 있다는 것이다.  

만약 데이터가 사람이 넣어주는 것이 아니라 파일에서 긁어오게 되는거라면, 더 이상 긁어 올 데이터가 없다는 뜻으로 EOF오류가 발생하게 된다.  

이런 경우에는 EOF오류를 예외처리 해주면 된다.  

~~~
while True:
    try:
        tmp = input()
        print(tmp)
    except EORError:
        break
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
