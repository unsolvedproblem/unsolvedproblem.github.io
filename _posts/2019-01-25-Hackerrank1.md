---
layout: post
title:  "Hackerrank 레벨업 하기46 [Jumping on the Clouds]"
date:   2019-01-25
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Jumping on the Clouds](https://www.hackerrank.com/challenges/jumping-on-the-clouds/problem)

~~~
def jumpingOnClouds(c):
    location = 0
    check = 0
    while True:
        if location + 2 == len(c) - 1:
            check += 1
            break
        elif location + 1 == len(c) - 1:
            check += 1
            break
        elif c[location + 2] == 0:
            location += 2
        else:
            location += 1
        check += 1
    return check
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
