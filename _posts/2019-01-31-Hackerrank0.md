---
layout: post
title:  "Hackerrank 30 Days of Code 5 [Let's Review]"
date:   2019-01-31
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Let's Review](https://www.hackerrank.com/challenges/30-review-loop/problem)

~~~
case_number =int(input())
strs = []
for i in range(case_number):
    strs.append(input())

def get_even_odd(string):
    even = ''
    odd = ''
    for i in range(len(string)):
        if i % 2 == 0:
            even += string[i]
        else:
            odd += string[i]
    print(even, odd)

if __name__=='__main__':
    for i in strs:
        get_even_odd(i)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
