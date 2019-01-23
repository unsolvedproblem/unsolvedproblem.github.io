---
layout: post
title:  "Hackerrank 레벨업 하기36 [Sequence Equation]"
date:   2019-01-23
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Sequence Equation](https://www.hackerrank.com/challenges/permutation-equation/problem)

~~~
def permutationEquation(p):
    result = []
    for i in range(len(p)):
        result.append(p.index(p.index(i+1)+1) + 1)
        print(result)
    return result
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
