---
layout: post
title:  "Hackerank 레벨업 하기12"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Kangaroo](https://www.hackerrank.com/challenges/kangaroo/problem)

~~~
def kangaroo(x1, v1, x2, v2):
    if x2 < x1:
        a1 = x2
        vv1 = v2
        a2 = x1
        vv2 = v1
    else:
        a1 = x1
        vv1 = v1
        a2 = x2
        vv2 = v2
    if vv2 >=  vv1:
        return "NO"
    elif (a2-a1) % (vv2 - vv1) == 0:
        return "YES"
    else:
        return "NO"
~~~

궁금하신게 있다면 메일 주세요.
k2h7913@daum.net  
cafehero123@gmail.com
