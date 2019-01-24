---
layout: post
title:  "Hackerrank 레벨업 하기41 [Sherlock and Squares]"
date:   2019-01-24
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Sherlock and Squares](https://www.hackerrank.com/challenges/sherlock-and-squares/problem)

~~~
def squares(a, b):
    if a ** (1/2) != int(a ** (1/2)):
        return len(list(range(int(a ** (1/2)) + 1, int(b ** (1/2))+ 1 )))
    else:
        return len(list(range(int(a ** (1/2)) ,int(b ** (1/2)) + 1)))
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
