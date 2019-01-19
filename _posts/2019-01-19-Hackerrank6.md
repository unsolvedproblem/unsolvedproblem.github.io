---
layout: post
title:  "Hackerank 레벨업 하기17"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Migratory Birds](https://www.hackerrank.com/challenges/migratory-birds/problem)

~~~
def migratoryBirds(arr):
    num1, num2, num3, num4, num5 = 0, 0, 0, 0, 0
    for i in arr:
        if i == 1: num1 +=1
        elif i == 2: num2 +=1
        elif i == 3: num3 +=1
        elif i == 4: num4 +=1
        else: num5 +=1
    tmp = [num1 ,num2, num3, num4, num5]
    return tmp.index(max(tmp)) + 1
~~~

궁금하신게 있다면 메일 주세요.
k2h7913@daum.net  
cafehero123@gmail.com
