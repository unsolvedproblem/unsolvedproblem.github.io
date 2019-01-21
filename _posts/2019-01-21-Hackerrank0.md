---
layout: post
title:  "Hackerank 레벨업 하기25 [Forming a Magic Square]"
date:   2019-01-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Forming a Magic Square](https://www.hackerrank.com/challenges/magic-square-forming/problem)

~~~
def formingMagicSquare(s):
    s1 = [[8,3,4],[1,5,9],[6,7,2]]
    s2 = [[4,9,2],[3,5,7],[8,1,6]]
    s3 = [[2,7,6],[9,5,1],[4,3,8]]
    s4 = [[6,1,8],[7,5,3],[2,9,4]]
    s5 = [[6,7,2],[1,5,9],[8,3,4]]
    s6 = [[2,9,4],[7,5,3],[6,1,8]]
    s7 = [[4,3,8],[9,5,1],[2,7,6]]
    s8 = [[8,1,6],[3,5,7],[4,9,2]]
    magic_sq =[s1,s2,s3,s4,s5,s6,s7,s8]
    cost = []
    for i in range(len(magic_sq)):
        respective_cost = 0
        for j in range(3):
            for k in range(3):
                respective_cost += abs(s[j][k] - magic_sq[i][j][k])
        cost.append(respective_cost)
    return min(cost)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
