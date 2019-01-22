---
layout: post
title:  "Hackerrank 레벨업 하기33 [Viral Advertising]"
date:   2019-01-22
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Viral Advertising](https://www.hackerrank.com/challenges/strange-advertising/problem)

~~~
def viralAdvertising(n):
    share = 5
    like = 2
    cumulative = 2
    for i in range(n-1):
        share = 3 * like
        like = int(share/2)
        cumulative += like
        print(share, like, cumulative)
    return cumulative
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
