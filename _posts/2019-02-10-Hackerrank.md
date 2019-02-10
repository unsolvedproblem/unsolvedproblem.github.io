---
layout: post
title:  "Hackerrank 30 Days of Code 11 [2D Arrays]"
date:   2019-02-10
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[2D Arrays](https://www.hackerrank.com/challenges/30-2d-arrays/problem)

~~~
def search_hourglasses(arr):
    i = len(arr)
    j = len(arr[0])

    if i < 2 or j < 2:
        return 0

    check_point = []
    sum_hourglasses = []
    for m in range(i - 2):
        for n in range(j - 2):
            tmp = arr[m][n] + arr[m][n+1] + arr[m][n+2] + arr[m+1][n+1] + arr[m+2][n] + arr[m+2][n+1] + arr[m+2][n+2]
            sum_hourglasses.append(tmp)
    return max(sum_hourglasses)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
