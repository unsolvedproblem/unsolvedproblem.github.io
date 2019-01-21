---
layout: post
title:  "Hackerrank 레벨업 하기24 [Cats and a Mouse]"
date:   2019-01-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Cats and a Mouse](https://www.hackerrank.com/challenges/cats-and-a-mouse/problem)

~~~
def catAndMouse(x, y, z):
    dist = [abs(x-z), abs(y-z)]
    if dist[0] == dist[1]:
        return 'Mouse C'
    else:
        if dist.index(min(dist)) == 0:
            return 'Cat A'
        else:
            return 'Cat B'
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
