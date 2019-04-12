---
layout: post
title:  "사다리 조작 삼성 coding test preparing7 python3 파이썬3"
date:   2019-04-12
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[사다리 조작](https://www.acmicpc.net/problem/15684)

~~~
import sys
n, m, h = list(map(int, sys.stdin.readline().split()))
bars = []
for i in range(m):
    tmp = list(map(int, sys.stdin.readline().split()))
    bars.append(tmp)
board = [[0] * n for _ in range(h)]
for bar in bars:
    x, y = bar[0], bar[1]
    board[x-1][y-1] = 1
    board[x-1][y] = 2
## 내려가기
def down():
    #print(board, 'down')
    for i in range(n):
        location = i
        for j in range(h):
            if board[j][location] == 1:
                location += 1
            elif board[j][location] == 2:
                location -= 1
        if location != i:
            return False
    return True

success = False
## 출력
if down() == True:
    success = True
    print(0)
else:
    ## dfs
    flat_board = []
    for i in board:
        flat_board += i
    # print(flat_board, 'flat_board')
    hope_ = []
    from collections import deque
    visit = deque([])
    def hope(arr, m, i=0):
        global board, success, hope
        if len(visit) == m:
            return
        elif arr == []:
            return
        else:
            for t in range(i, len(arr)-1):
                if (t+1) % n == 0: continue
                height = t // n
                tmp = t % n
                if board[height][tmp] == 0 and board[height][tmp+1] == 0:
                    visit.append((height, tmp))
                    board[height][tmp] = 1
                    board[height][tmp+1] = 2
                    #print(visit, 'visit')
                    #print(board, 'board')
                    #print(down())
                    if down() == True:
                        success = True
                        hope_.append(len(visit))
                        #print(board, 'final_visit')
                        #print(len(visit))
                    hope(arr, m, i=t+1)
                    board[height][tmp] = 0
                    board[height][tmp+1] = 0
                    visit.pop()
    hope(flat_board, 3)
    if success == True:
        #print(hope_)
        print(min(hope_))
if success == False:
    print(-1)
~~~
시간 초과의 문제가 계속 발생해서 여러가지 조작을 했습니다.

1. 읽어오는 속도를 향상 시키기 위해 input() 대신 sys.stdin.readline()을 사용했습니다.
2. pop()의 속도를 향상 시키기 위해 collections에 있는 deque 클래스를 사용했습니다.


이 문제에서는 dfs를 사용하는게 가장 중요했습니다. 하지만 저는 2차원 리스트에서 dfs를 사용하는 방법을 몰라서 2차원인 board를  1차원인 flat_board로 만들어서 해결하였습니다. dfs를 만들 때 참고한 방법은 조합을 만드는 방법입니다.

~~~
def combination(arr, m):
    visit = []
    def generator(arr, m, i=0):
        if len(visit) == m:
            print(visit)
            return
        else:
            for j in range(i, len(arr)):
                visit.append(arr[j])
                generator(arr, m, j+1)
                visit.pop()
    generator(arr, m)
~~~






궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
