---
layout: post
title:  "Hackerank 레벨업 하기4"
date:   2019-01-18
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---
Hackerrank 레벨업하기  
문제  
[Diagonal Difference](https://www.hackerrank.com/challenges/diagonal-difference/problem)

~~~
def diagonalDifference(arr):
    a, b = 0, 0
    n = len(arr)
    for i in range(len(arr)):
        a += arr[i][i]
        b += arr[n-1 -i][i]
    return abs(a-b)
~~~

궁금하신게 있다면 메일 주세요.   
k2h7913@daum.net  
cafehero123@gmail.com
