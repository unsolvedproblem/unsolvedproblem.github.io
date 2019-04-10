---
layout: post
title:  "치킨 배달 삼성 coding test preparing5 python3 파이썬3"
date:   2019-04-10
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[치킨 배달](https://www.acmicpc.net/problem/15686)

~~~
from itertools import  combinations

n, m  = list(map(int, input().split()))
board = []
for _ in range(n):
    tmp = list(map(int, input().split()))
    board.append(tmp)
## 전처리
houses = []
chicken = []
for i in range(n):
    for j in range(n):
        if board[i][j] == 1: houses.append((i, j))
        elif board[i][j] == 2: chicken.append((i,j))
combi = []
for i in combinations(chicken, m):
    combi.append(i)
#print(combi)
#print(houses)
## search
def search():
    global result
    result = []
    for chick in combi:
        city_chick_length = 0
        for house in houses:
            x, y = house[0], house[1]
            chick_length = []
            for i, j in chick:
                chick_length.append(abs(x-i) + abs(y-j))
            #print(chick_length, 'asdf')
            city_chick_length += min(chick_length)
        result.append(city_chick_length)

search()
#print(result, 'asdfsadfsadfsadf')
print(min(result))
~~~








궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
