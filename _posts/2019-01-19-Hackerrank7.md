---
layout: post
title:  "Hackerank 레벨업 하기18"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Day of the Programmer](https://www.hackerrank.com/challenges/day-of-the-programmer/problem)

~~~
def dayOfProgrammer(year):
    if 1700 <= year and year <= 1917:
        if year % 4 ==0:
            return '12.09.%s' %year
        else:
            return '13.09.%s'  %year

    elif year == 1918:
        return '26.09.%s' %year

    else:
        if year % 400 == 0:
            return '12.09.%s' %year
        elif year % 4 == 0 and year % 100 != 0:
            return '12.09.%s' %year
        else:
            return '13.09.%s' %year
~~~

궁금하신게 있다면 메일 주세요.
