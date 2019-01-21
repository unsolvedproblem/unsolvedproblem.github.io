---
layout: post
title:  "Hackerrank 레벨업 하기27 [Climbing the Leaderboard]"
date:   2019-01-20
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Climbing the Leaderboard](https://www.hackerrank.com/challenges/climbing-the-leaderboard/problem)

~~~
def climbingLeaderboard(scores, alice):
    rank = []
    rescores = sorted(set(scores), reverse=True)
    l = len(rescores)
    t = 0
    for i in alice:
        if i < rescores[-1] :
            rank.append(l+1)
        elif rescores[0] <= i:
            rank.append(1)
        else:
            for j in range(t, l - 1):
                if rescores[l - 1 - j] <= i < rescores[l - 1 -j - 1]:
                    rank.append(l - j )
                    t = j
                    break
    return rank
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
