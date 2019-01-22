---
layout: post
title:  "Hackerrank 레벨업 하기34 [Save the Prisoner]"
date:   2019-01-22
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Save the Prisoner](https://www.hackerrank.com/challenges/save-the-prisoner/problem)

~~~
def saveThePrisoner(n, m, s):
    # n = # prisoners
    # m = # sweets
    # s = # starting point
    if (m + s - 1) % n == 0:
        return n
    else:
        return((m + s - 1) %  n) 
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
