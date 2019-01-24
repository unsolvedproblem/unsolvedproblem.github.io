---
layout: post
title:  "Hackerrank 레벨업 하기44 [Non-Divisible Subset]"
date:   2019-01-24
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Non-Divisible Subset](https://www.hackerrank.com/challenges/non-divisible-subset/problem)

~~~
def nonDivisibleSubset(k, S):
    s = list(map(lambda x: x % k,S))
    s.sort()
    ss = []
    num = 0
    while True:
        ss.append([])
        tmp = s[0]
        while tmp in s:
            ss[num].append(s.pop(s.index(tmp)))
        if s == []:
            break
        num += 1
    result = 0
    for i in range(len(ss)):
        if ss[i][0] == 0:
            result += 1
        else:
            if k / ss[i][0] == 2:
                result += 1
            else:
                result += len(ss[i])
            for j in range(i):
                if ss[j][0] + ss[i][0] == k:
                    result = result - len(ss[j]) - len(ss[i])
                    result = result + max(len(ss[i]),len(ss[j]))
                    break
    return result
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
