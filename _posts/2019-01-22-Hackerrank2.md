---
layout: post
title:  "Hackerrank 레벨업 하기32 [Beautiful Days at the Movies]"
date:   2019-01-22
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Beautiful Days at the Movies](https://www.hackerrank.com/challenges/beautiful-days-at-the-movies/problem)

~~~
def beautifulDays(i, j, k):
    count = 0
    days = list(range(i, j+1))
    for i in days:
        rev = list(str(i))
        rev.reverse()
        diff = abs(int(''.join(rev)) - i)
        if diff % k ==0:
            count += 1
    return count
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
