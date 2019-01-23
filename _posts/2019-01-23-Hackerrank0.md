---
layout: post
title:  "Hackerrank 레벨업 하기35 [Circular Array Rotation]"
date:   2019-01-23
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Circular Array Rotation](https://www.hackerrank.com/challenges/circular-array-rotation/problem)

~~~
def circularArrayRotation(a, k, queries):
    #a = array
    #k = # rotation
    #queries = indice
    k = k % len(a)
    a = a[-k:] + a[:-k]
    result = []
    for i in queries:
        result.append(a[i])
    return result
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
