---
layout: post
title:  "Hackerank 레벨업 하기9"
date:   2019-01-18
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Time Conversion](https://www.hackerrank.com/challenges/time-conversion/problem)

~~~
def timeConversion(s):
    AM_PM  = s[-2:]
    TIME0 = s[:-2]
    time  = TIME0.split(':')
    if AM_PM == 'AM':
        if time[0] == '12':
            time[0] = '00'
        return  ':'.join(time)
    else:
        if time[0] !='12':
            time[0] = str(int(time[0])+12)
        return ':'.join(time)
~~~

궁금하신게 있다면 메일 주세요.   
k2h7913@daum.net  
cafehero123@gmail.com
