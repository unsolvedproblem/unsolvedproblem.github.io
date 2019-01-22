---
layout: post
title:  "Hackerrank 레벨업 하기31 [Angry Professor]"
date:   2019-01-22
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Angry Professor](https://www.hackerrank.com/challenges/angry-professor/problem)

~~~
def angryProfessor(k, a):
    watchout = [x <= 0 for x in a]
    if watchout.count(1) < k:
        return 'YES'
    else:
        return 'NO'
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
