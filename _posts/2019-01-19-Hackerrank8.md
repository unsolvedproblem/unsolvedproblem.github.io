---
layout: post
title:  "Hackerrank 레벨업 하기19 [Bon Appetit]"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Bon Appetit](https://www.hackerrank.com/challenges/bon-appetit/problem)

~~~
def bonAppetit(bill, k, b):
    del bill[k]
    b_actual = sum(bill)/2
    if b == b_actual:
        print('Bon Appetit')
    else:
        print(int(b - b_actual))
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
