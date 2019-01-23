---
layout: post
title:  "Hackerrank 레벨업 하기38 [Find Digits]"
date:   2019-01-23
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Find Digits](https://www.hackerrank.com/challenges/find-digits/problem)

~~~
def findDigits(n):
    string = list(str(n))
    num = 0
    for i in string:
        if i == '0':
            pass
        elif n % int(i) == 0:
            num += 1
    return num
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
