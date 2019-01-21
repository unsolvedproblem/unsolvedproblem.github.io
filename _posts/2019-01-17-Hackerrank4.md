---
layout: post
title:  "Hackerrank 레벨업 하기5 [Plus Minus]"
date:   2019-01-18
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Plus Minus](https://www.hackerrank.com/challenges/plus-minus/problem)

~~~
def plusMinus(arr):
    a, b, c = 0,0,0
    n=len(arr)
    for i in arr:
        if i >0:
            a +=1
        elif i <0:
            b +=1
        else:
            c+=1
    print(a/n)
    print(b/n)
    print(c/n)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
