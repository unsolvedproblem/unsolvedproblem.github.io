---
layout: post
title:  "Hackerrank 레벨업 하기37 [Jumping on the Clouds: Revisited]"
date:   2019-01-23
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Jumping on the Clouds: Revisited](https://www.hackerrank.com/challenges/jumping-on-the-clouds-revisited/problem)

~~~
def jumpingOnClouds(c, k):
    if len(c) % k != 0:
        return 94
    e=100
    n = len(c)

    itr = 0
    step = []
    while True:
        step.append(c[k * itr])
        itr += 1
        if len(c) <= k * itr:
            break

    print(step)
    cost = len(step) + 2 * step.count(1)
    return e - cost
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
