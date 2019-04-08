---
layout: post
title:  "인구 이동 삼성 coding test preparing4 python3 파이썬3"
date:   2019-04-08
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[인구 이동](https://www.acmicpc.net/problem/16234)

~~~
n, l, r = list(map(int, input().split()))
board = []
for i in range(n):
    tmp = list(map(int, input().split()))
    board.append(tmp)
## search
def search():
    global q
    compare = (-1, 0), (0, -1), (1, 0), (0, 1)
    q = []
    for x in range(n):
        for y in range(n):
            for dx, dy in compare:
                nx, ny = x + dx, y + dy
                if nx < 0 or n <= nx: continue
                if ny < 0 or n <= ny: continue
                if l <= abs(board[x][y] - board[nx][ny]) <=  r:
                    if ((nx, ny), (x, y)) not in q:
                        q.append(((x, y),(nx, ny)))
    #print(q)
    return
## grouping
def grouping_fnc():
    global grouping

    grouping = []
    #num1 = 0

    while q:
        group_q = []

        cur, nei = q.pop(0)
        group_q.append(cur)
        group_q.append(nei)

        while True:
            tmp = len(q)
            del_index = []
            for k in range(len(q)):
                if q[k][0] in group_q:
                    if q[k][1] not in group_q:
                        group_q.append(q[k][1])
                    del_index = [k] + del_index
                elif q[k][1] in group_q:
                    if q[k][0] not in group_q:
                        group_q.append(q[k][0])
                    del_index = [k] + del_index
            #print(group_q, del_index, '      middle     ', q)
            for i in del_index:
                del q[i]
            #print(group_q, del_index, '      middle     ', q)
            if tmp == len(q):
                break
        #print(group_q, del_index, '      middle     ', q)
        grouping.append(group_q)

        #print(grouping, '       end         ',group_q)
    #print(grouping, 'grouping')
    return
def update():
    if grouping != []:
        for group in grouping:
            mean = 0
            n_countries = len(group)
            for country in group:
                x, y = country[0], country[1]
                mean +=  board[x][y]
            mean = int(mean/ n_countries)
            for country in group:
                x, y = country[0], country[1]
                board[x][y] = mean
    #print(board)
    return
num = 0
while True:
    search()
    if q == []:
        break
    grouping_fnc()
    update()
    num += 1
print(num)
~~~








궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
