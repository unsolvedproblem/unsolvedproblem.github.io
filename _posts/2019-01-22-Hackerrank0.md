---
layout: post
title:  "Hackerrank 레벨업 하기30 [Utopian Tree]"
date:   2019-01-22
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Utopian Tree](https://www.hackerrank.com/challenges/utopian-tree/problem)

~~~
def utopianTree(n):
    t = 1
    for i in range(n):
        if i % 2 == 0:
            t = t * 2
        else:
            t += 1
    return t
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
