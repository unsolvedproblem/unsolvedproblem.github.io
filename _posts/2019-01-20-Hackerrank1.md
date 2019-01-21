---
layout: post
title:  "Hackerank 레벨업 하기22 [Counting Valleys]"
date:   2019-01-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Counting Valleys](https://www.hackerrank.com/challenges/counting-valleys/problem)

~~~
def countingValleys(n, s):
    s = list(s)
    level = 0
    points = []
    for i in s:
        if i =='U':
            level += 1
        else:
            level += -1
        points.append(level)
    count = 0
    for i in range(len(points) -1):
        if points[i] < 0 and points[i+1] ==0:
            count +=1
    return count
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
