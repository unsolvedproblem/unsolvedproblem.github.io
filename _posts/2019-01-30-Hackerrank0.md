---
layout: post
title:  "Hackerrank 30 Days of Code 4 [Class vs. Instance]"
date:   2019-01-30
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Class vs. Instance](https://www.hackerrank.com/challenges/30-class-vs-instance/problem)

~~~
class Person:
    def __init__(self, initialAge):
        self.age = initialAge
        if self.age < 0:
            print('Age is not valid, setting age to 0.')
            self.age = 0

    def yearPasses(self):
        self.age += 1

    def amIOld(self):
        if self.age < 13:
            print("You are young.")
        elif self.age >= 13 and self.age < 18:
            print("You are a teenager.")
        else:
            print("You are old.")
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
