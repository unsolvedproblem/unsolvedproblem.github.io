---
layout: post
title:  "삼성 coding test preparing2"
date:   2019-04-05
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[보물상자 비밀번호](https://www.acmicpc.net/problem/16236)

~~~
N = int(input())
map_=[]
for i in range(N):
    tmp = list(map(int, input().split()))
    map_.append(tmp)


## 아기상어 위치 찾기
q = []
d = 0
def init():
    global d, map_
    for i in range(N):
        for j in range(N):
            if map_[i][j] == 9:
                q.append((0, i, j)) ## 이동거리, 좌표
                map_[i][j] = 0
                return
init()    

### move
body = 2
eat = 0
time = 0
check = [[False] * N for _ in range(N)]
while q:
    ##이동
    d, x, y = q.pop(0)
    check[x][y] = True
    for dx, dy in (-1, 0), (0, -1), (1, 0), (0, 1):
        nd, nx, ny = d + 1, x + dx, y + dy
        if nx < 0 or nx >= N or ny < 0 or ny >= N:
            continue
        if map_[nx][ny] > body or check[nx][ny] == True:
            continue
    ## 큐 확장
        check[nx][ny] = True
        q.append((nd, nx, ny))
    if q!= [] and d == q[0][0]:
        continue
    ### 먹을게 있을 때!!!
    tmp1 = sorted(q, key=lambda x: (x[1], x[2]), reverse=False)
    for i in tmp1:
        d, nx, ny = i
        if 0 < map_[nx][ny] < body:
            eat += 1
            map_[nx][ny] = 9
            if eat == body:
                body += 1
                eat = 0
            time += d
            d = 0
            q = []
            check = [[False] * N for _ in range(N)]
            init()
            break

print(time)

~~~










궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
