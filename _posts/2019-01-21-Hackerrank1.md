---
layout: post
title:  "Hackerank 레벨업 하기26 [Picking Numbers]"
date:   2019-01-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Picking Numbers](https://www.hackerrank.com/challenges/picking-numbers/problem)

~~~
def pickingNumbers(a):
    collect = []
    for i in range(len(a)):
        collect.append([])
        for j in a:
            if 0 <= a[i] - j <= 1:
                collect[i].append(j)
    count = []
    for i in collect:
        count.append(len(i))
    return max(count)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
