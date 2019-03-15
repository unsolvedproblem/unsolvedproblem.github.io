---
layout: post
title:  "Hackerrank 30 Days of Code 20 [Sorting]"
date:   2019-03-15
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Sorting](https://www.hackerrank.com/challenges/30-sorting/problem)

~~~
swaps = 0

for i in range(n):
    for j in range(0, n - i - 1):
        if a[j] > a[j + 1]:
            a[j], a[j + 1] = a[j + 1], a[j]
            swaps += 1

print("Array is sorted in %s swaps."  %swaps)
print("First Element: %s" %a[0])
print("Last Element: %s" %a[-1])
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
