---
layout: post
title:  "Hackerank 레벨업 하기2"
date:   2019-01-18
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Compare the Triplets](https://www.hackerrank.com/challenges/compare-the-triplets/problem)

~~~
def compareTriplets(a, b):
    point = [0, 0]
    for i in range(len(a)):
        if a[i] > b[i]:
            point[0] +=1
        elif a[i] < b[i]:
            point[1] +=1
        else:
            pass
    return point
~~~

궁금하신게 있다면 메일 주세요.
