---
layout: post
title:  "Hackerrank 레벨업 하기47 [Equalize the Array]"
date:   2019-01-25
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Equalize the Array](https://www.hackerrank.com/challenges/equality-in-a-array/problem)

~~~
def equalizeArray(arr):
    t = list(set(arr))
    z = []
    for i in t:
        z.append(arr.count(i))
    z.sort()
    return sum(z[:-1])
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
