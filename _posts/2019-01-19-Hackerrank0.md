---
layout: post
title:  "Hackerank 레벨업 하기11"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Apple and Orange](https://www.hackerrank.com/challenges/apple-and-orange/problem)

~~~
def countApplesAndOranges(s, t, a, b, apples, oranges):
    num_app = 0
    num_orn = 0
    apples_loc = list(map(lambda x: x + a, apples))
    oranges_loc = list(map(lambda x: x + b, oranges))

    for i in apples_loc:
        if s <= i and i <= t:
            num_app += 1
    print(num_app)
    for i in oranges_loc:
        if s <= i and i <= t:
            num_orn += 1
    print(num_orn)
~~~

궁금하신게 있다면 메일 주세요.
