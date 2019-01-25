---
layout: post
title:  "Hackerrank 레벨업 하기45 [Repeated String]"
date:   2019-01-25
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Repeated String](https://www.hackerrank.com/challenges/repeated-string/problem)

~~~
def repeatedString(s, n):
    t = len(s)
    a_num = s.count('a')
    div = n // t
    print(div)
    mod = n % t
    print(mod)
    return div * a_num + s[:mod].count('a')
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
