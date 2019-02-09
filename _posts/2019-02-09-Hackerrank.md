---
layout: post
title:  "Hackerrank 30 Days of Code 10 [Binary Numbers]"
date:   2019-02-09
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Binary Numbers](https://www.hackerrank.com/challenges/30-binary-numbers/problem)

~~~
n = int(input())
binary_number = bin(n)
count = 0
arr = []
for i in binary_number:
    if i == '1':
        count += 1
    else:
        count = 0
    arr.append(count)

print(max(arr))
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
