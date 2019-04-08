---
layout: post
title:  "삼성 coding test preparing3 python3 파이썬3"
date:   2019-04-06
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[아기상어](https://www.acmicpc.net/problem/16236)

~~~
n = int(input())
board=[]
for i in range(n):
    tmp = list(map(int, input().split()))
    board.append(tmp)

class shark:
    def __init__(self):
        self.body = 2
        self.eat = 0
        self.time = 0
    def init(self):
        global board
        for i in range(n):
            for j in range(n):
                if board[i][j] == 9:
                    tmp = (i, j, 0)
                    board[i][j] = 0
                    return tmp
shark = shark()
state = shark.init()
move = (-1, 0), (0, -1), (0, 1), (1, 0)

q = []
q.append(state)
check = [[False] * n for _ in range(n)]
# num = 0 #########debug
while q:
    #print(q, 'top')
    x, y, d = q.pop(0)
    check[x][y] = True

    for i, j in move:
        if not 0 <= x+i < n: continue
        if not 0 <= y+j < n: continue
        if shark.body < board[x+i][y+j]: continue
        if check[x+i][y+j] == True: continue
        if (x + i, y + j, d + 1) not in q:
            q.append((x + i, y + j, d + 1))

    #print(q, 'middle')
    if q==[]:
        break
    elif q[0][2] == d:
        continue

    #let's eat
    #print(q, 'bottom')
    q_sort = sorted(q, key=lambda x:  (x[0], x[1]), reverse=False)
    for i, j, spended_time in q_sort:
        if 0 < board[i][j] < shark.body:
            print('q_sort', q_sort)
            shark.eat += 1
            shark.time += spended_time
            if shark.eat == shark.body:
                shark.body += 1
                shark.eat = 0
            board[i][j] = 9
            q=[]
            q.append(shark.init())
            check = [[False] * n for _ in range(n)]
            break
    #num += 1
    #if num > 10: break
print(shark.time)
~~~
~~~
x = [[1,2,3],[2,3,1],[3,2,1]]
sorted(x)
sorted(x, key=lambda order: order[0])
sorted(x, key=lambda order: order[1])
sorted(x, key=lambda order: order[2])

sorted(x, key=lambda order: (order[0],order[1]))
sorted(x, key=lambda order: (order[0],order[1]), reverse=True)
sorted(x, key=lambda order: (order[0],order[1]), reverse=False)
~~~








궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
