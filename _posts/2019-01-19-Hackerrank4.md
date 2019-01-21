---
layout: post
title:  "Hackerrank 레벨업 하기15 [Birthday Chocolate]"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Birthday Chocolate](https://www.hackerrank.com/challenges/the-birthday-bar/problem)

~~~
def birthday(s, d, m):
    num = 0
    for i in range(len(s)+1 - m):
        if sum(s[i:i+m]) == d:
            num += 1
    return num
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
