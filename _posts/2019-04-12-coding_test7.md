---
layout: post
title:  "coding test preparing DFS BFS python3 파이썬3"
date:   2019-04-13
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

1. bfs, dfs: queue와 stack에 들어가기위한 조건 잘 확인하기   
2. '+'는 append 보다 느림
3. if-else문보다 continue를 걸어주는게 빠름
4. deque + while + pop이 for문보다 빠름




## BFS
아기상어
~~~
############# bfs
import sys

n = int(sys.stdin.readline())
board = []
for _ in range(n):
    tmp = list(map(int, sys.stdin.readline().split()))
    board.append(tmp)
moves = ((1, 0),(-1, 0),(0, 1),(0, -1))
shark_body = 2
shark_eat = 0
shark_dis = 0
def init():
    global shark_location
    global queue
    global check
    queue = []
    check = [[False] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            if board[i][j] == 9:
                shark_location = (i, j)
                board[i][j] = 0
                check[i][j] = True
                queue.append((i, j, 0))
                break
init()
## print(board,'\n', check, '\n',queue, 'top') ###########
while queue:
#### bfs
    x, y, d = queue.pop(0)
    ##### q.append
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if nx < 0: continue
        if n <= nx: continue
        if ny < 0: continue
        if n <= ny: continue
        if board[nx][ny] > shark_body: continue
        if check[nx][ny] == True: continue
        queue.append((nx, ny, d + 1))
        check[nx][ny] = True
    if queue == []:
        pass
    elif d == queue[0][2]:
        continue
    ## print(queue, '\n', check, '\n', board, 'middle')
    #### eat!!!
    tmp_que = sorted(queue, key=lambda x: (x[0], x[1]))
    for now_x, now_y, now_d in tmp_que:
        if 0 < board[now_x][now_y] < shark_body:
            shark_eat += 1
            shark_dis += now_d
            board[now_x][now_y] = 9
            if shark_eat == shark_body:
                shark_body += 1
                shark_eat = 0
            init()
            break
print(shark_dis)
~~~

인구
~~~
import sys
from collections import deque
n, l, r = list(map(int, sys.stdin.readline().split()))
board = []
for _ in range(n):
    tmp = list(map(int, sys.stdin.readline().split()))
    board.append(tmp)
moves = (1, 0), (0, 1), (-1, 0), (0, -1)

def init(i=0, j=0):
    global queue
    queue = deque([])
    queue.append((i,j))
    check[i][j] = True
def bfs():
    global check
    check = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if check[i][j] != False: continue
            init(i, j)
            sum_ = board[i][j]
            locations = deque([(i, j)])
            while queue:
                x, y = queue.popleft()
                #### 조건
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if nx < 0: continue
                    if n <= nx: continue
                    if ny < 0: continue
                    if n <= ny: continue
                    if check[nx][ny] != False: continue
                    if l <= abs(board[x][y] - board[nx][ny]) <= r:
                        queue.append((nx, ny))
                        locations.append((nx, ny))
                        sum_ += board[nx][ny]
                        check[nx][ny] = True
            mean = sum_ // len(locations)
            while locations:
                x, y = locations.popleft()
                check[x][y] = mean
    return check

count = 0

while True:
    tmp = bfs()
    if tmp == board:
        break
    count += 1
    board = tmp
print(count)
~~~


## DFS
CCTV
~~~
a = [4,2,3,2,1,1]
t = len(a)
b = [1] * 8
for i in range(t):
    b[i] = a[i]

candi = []
for i0 in range(b[0]):
    for i1 in range(b[1]):
        for i2 in range(b[2]):
            for i3 in range(b[3]):
                for i4 in range(b[4]):
                    for i5 in range(b[5]):
                        for i6 in range(b[6]):
                            for i7 in range(b[7]):
                                tmp = (i0,i1, i2, i3, i4, i5, i6, i7)
                                print(tmp[:t])
~~~

사다리
~~~
import sys
from collections import deque

n, m, h = list(map(int, sys.stdin.readline().split()))
bars = deque([])
for _ in range(m):
    tmp = list(map(int, sys.stdin.readline().split()))
    bars.append((tmp[0]-1, tmp[1]-1))

board = [[0] * n for _ in range(h)]
while bars:
    x, y = bars.pop()
    board[x][y] = 1
    board[x][y+1] = 2

#print(n, m, h)
#print(board)
#print('###########################top####################')

## 내려가기
def down(board):
    for j in range(n):
        location = j
        for i in range(h):
            if board[i][location] == 0:
                continue
            if board[i][location] == 1:
                location += 1
                continue
            if board[i][location] == 2:
                location -= 1
                continue
        if j != location:
            return False
    return True

def dfs(board):
    global count
    visit = deque([])
    count = 0
    def generator(limit=3, i=0):
        global count
        if len(visit) == limit:
            return
        else:
            for j in range(i, (n-1)*h):
                x = j // (n-1)
                y = j % (n-1)
                ## 조건
                if board[x][y] != 0: continue
                if board[x][y+1] != 0: continue
                visit.append((x,y))
                board[x][y] = 1
                board[x][y+1] = 2
                #print(board)
                tmp = down(board)
                if tmp:
                    if len(visit) <= limit:
                        limit = len(visit)
                        count = limit
                generator(limit, i=j+1)
                visit.pop()
                board[x][y] = 0
                board[x][y+1] = 0
    generator()
if down(board):
    print(0)
else:
    dfs(board)
    if count:
        print(count)
    else:
        print(-1)
~~~
