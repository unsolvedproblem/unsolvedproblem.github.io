---
layout: post
title:  "Hackerrank 30 Days of Code 14 [Scope]"
date:   2019-02-13
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Scope](https://www.hackerrank.com/challenges/30-scope/problem)

~~~
def computeDifference(self):
    max = 0
    for i in range(len(self.__elements)):
        for j in range(i, len(self.__elements)):
            if max <= abs(self.__elements[i] - self.__elements[j]):
                max = abs(self.__elements[i] - self.__elements[j])

    self.maximumDifference = max
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
