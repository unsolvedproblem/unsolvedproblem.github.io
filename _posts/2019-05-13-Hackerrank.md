---
layout: post
title:  "Hackerrank 30 Days of Code 24 [Running Time and Complexity]"
date:   2019-03-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Running Time and Complexity](https://www.hackerrank.com/challenges/30-running-time-and-complexity/problem)

~~~
def primality(n):
    import math

    if n == 1 or n == 4:
        return "Not prime"
    if n == 2 or n == 3:
        return "Prime"

    for i in range(2, math.ceil(n ** (1/2)+1)):
        if n % i == 0:
            return "Not prime"

    return  "Prime"
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
