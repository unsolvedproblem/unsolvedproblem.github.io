---
layout: post
title:  "Hackerrank 레벨업 하기43 [Cut the sticks]"
date:   2019-01-24
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Cut the sticks](https://www.hackerrank.com/challenges/cut-the-sticks/problem)

~~~
def cutTheSticks(arr):
    result = []
    while True:
        result.append(len(arr))
        tmp = list(map(lambda x: x - min(arr), arr))
        arr = []
        for i in tmp:
            if i > 0:
                arr.append(i)
        if arr ==[]:
            break
    return result
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
