---
layout: post
title:  "Hackerrank 레벨업 하기23 [Electronics Shop]"
date:   2019-01-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Electronics Shop](https://www.hackerrank.com/challenges/electronics-shop/problem)

~~~
def getMoneySpent(keyboards, drives, b):
    expected_budget = []
    for i in keyboards:
        for j in drives:
            expected_budget.append(i+j)
    if min(expected_budget) > b:
        return -1
    else:
        can_buy = []
        for i in expected_budget:
            if i <= b:
                can_buy.append(i)
        return max(can_buy)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
