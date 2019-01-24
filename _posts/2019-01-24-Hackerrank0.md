---
layout: post
title:  "Hackerrank 레벨업 하기40 [Append and Delete]"
date:   2019-01-24
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Append and Delete](https://www.hackerrank.com/challenges/append-and-delete/problem)

~~~
def appendAndDelete(s, t, k):
    n_s = len(s)
    n_t = len(t)
    n_min = min(n_s, n_t)
    n_same = 0
    for i in range(n_min):
        if s[i] == t[i]:
            n_same += 1
        else:
            break
    n_s_diff = n_s - n_same
    n_t_diff = n_t - n_same

    if n_s + n_t <= k:
        return 'Yes'
    elif n_s_diff + n_t_diff <= k:
        if (n_s - n_s_diff) != 0:
            if ((k - n_s_diff) - n_t_diff) % 2 ==0:
                return 'Yes'
            else:
                return 'No'
        else:
            return 'Yes'
    else:
        return 'No'
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
