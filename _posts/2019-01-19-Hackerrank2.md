---
layout: post
title:  "Hackerank 레벨업 하기13 [Between Two Sets]"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Between Two Sets]https://www.hackerrank.com/challenges/between-two-sets/problem)

~~~
def getTotalX(a, b):
    def gcd(a,b):
        while (b!=0):
            tmp = a % b
            a = b
            b = tmp
        return abs(a)
    def lcm(a,b):
        gcd_value = gcd(a,b)
        return int((a * b)/gcd_value)
    if len(a) == 1:
        lcm1 = a[0]
    else:
        lcm1 = a[0]
        for i in range(1, len(a)):
            lcm1 = lcm(lcm1, a[i])

    div1 =[]
    num = 0
    while True:
        num += 1
        div1.append((lcm1 * num))
        if lcm1 * num >= b[0]:
           break
    num1 = 0
    for i in div1:
        TF= []
        for j in b:
            if j % i == 0:
                TF.append(1)
            else:
                TF.append(0)
        if TF == [1] * len(b):
            num1 += 1

    return num1
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
