---
layout: post
title:  "Hackerank 레벨업 하기21"
date:   2019-01-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Drawing Book](https://www.hackerrank.com/challenges/drawing-book/problem)

~~~
def pageCount(n, p):
    a = [x for x in range(n+1)]
    if len(a) % 2 !=0:
        a.append(0)
    num = 0
    pages = []
    while True:
        pages.append(a[num:num+2])
        num += 2
        if num > len(a) -1: break
    for i in pages:
        if p in i: page = i
    return min(pages.index(page), len(pages)-1 - pages.index(page))
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
