---
layout: post
title:  "Hackerank 레벨업 하기14"
date:   2019-01-19
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Breaking the Records](https://www.hackerrank.com/challenges/breaking-best-and-worst-records/problem)

~~~
def breakingRecords(scores):
    max_num = 0
    min_num = 0
    game = []
    game.append(scores[0])
    max_score = max(game)
    min_score = min(game)

    for i in range(1,len(scores)):
        game.append(scores[i])
        if max_score != max(game):
            max_num += 1
        elif min_score != min(game):
            min_num += 1
        else:
            pass
        max_score = max(game)
        min_score = min(game)
    return(max_num, min_num)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
