---
layout: post
title:  "Hackerank 레벨업 하기7"
date:   2019-01-18
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---
Hackerrank 레벨업하기  
문제  
[Mini-Max SUm](https://www.hackerrank.com/challenges/mini-max-sum/problem)

~~~
def miniMaxSum(arr):
    arr.sort()
    min_sum = 0
    max_sum = 0
    for i in range(4):
        min_sum += arr[i]
    for i in range(4):
        max_sum +=arr[1+i]

    print(min_sum, max_sum)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
