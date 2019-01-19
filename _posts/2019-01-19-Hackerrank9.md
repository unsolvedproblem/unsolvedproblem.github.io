---
layout: post
title:  "Hackerank 레벨업 하기20"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Sock Merchant](https://www.hackerrank.com/challenges/sock-merchant/problem)

~~~
def sockMerchant(n, ar):
    num = 0
    while ar:
        tmp = ar.pop(0)
        if tmp in ar:
            num += 1
            del ar[ar.index(tmp)]
    return num
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
