---
layout: post
title:  "Hackerrank 레벨업 하기49 [ACM ICPC Team]"
date:   2019-01-25
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[ACM ICPC Team](https://www.hackerrank.com/challenges/acm-icpc-team/problem)

~~~
def acmTeam(topic):
    for i in range(len(topic)):
        topic[i] = int('0b' + topic[i], 2)

    combination = []
    for i in range(len(topic) - 1):
        for j in range(i+1, len(topic)):
            combination.append(bin(topic[i]|topic[j]))

    count = []
    for i in combination:
        count.append(i.count('1'))

    return max(count), count.count(max(count))
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
