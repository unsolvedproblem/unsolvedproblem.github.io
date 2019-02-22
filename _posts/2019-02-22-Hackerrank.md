---
layout: post
title:  "Hackerrank 30 Days of Code 17 [More Exceptions]"
date:   2019-02-22
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[More Exceptions]https://www.hackerrank.com/challenges/30-more-exceptions/problem)

~~~
class MyError(Exception):
    def __str__(self):
        return "n and p should be non-negative"

class Calculator():
    def power(self, n,p):
        if n  < 0 or p < 0:
            raise MyError()
        else:
            return n ** p
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
