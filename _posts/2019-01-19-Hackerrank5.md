---
layout: post
title:  "Hackerank 레벨업 하기16"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Divisible Sum Pairs](https://www.hackerrank.com/challenges/divisible-sum-pairs/problem)

~~~
def divisibleSumPairs(n, k, ar):
    num = 0
    for i in range(len(ar)):
        for j in range(i+1, len(ar)):
            if (ar[i] + ar[j]) % k == 0:
                num += 1
    return num
~~~

궁금하신게 있다면 메일 주세요.
k2h7913@daum.net  
cafehero123@gmail.com
