---
layout: post
title:  "삼성 coding test preparing3"
date:   2019-04-06
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[나무 재테크](https://www.acmicpc.net/problem/16235)

~~~
n, m, k = list(map(int, input().split()))
board = [[5] * n for _ in range(n)]
foods = []
for i in range(n):
    tmp = list(map(int, input().split()))
    foods.append(tmp)

trees = []
for i in range(m):
    tmp = list(map(int, input().split()))
    trees.append((tmp[0] - 1, tmp[1] - 1, tmp[2]))

reproduce = (-1, -1),(-1, 0),(0, -1),(1, 1),(-1, 1),(1, -1),(0, 1),(1, 0)

#trees = sorted(trees, key=lambda x: x[2])

def spring():
    global trees
    global board
    global trees_remain, trees_dead

    trees_remain = []
    trees_dead = []
    for tree in trees:
        x, y, age = tree[0], tree[1], tree[2]
        if board[x][y] < age:
            trees_dead.append((x, y, int(age/2)))
            continue
        board[x][y] -= age
        age +=1
        trees_remain.append((x, y, age))

    #print(trees_remain, trees_dead)
    return trees_remain.copy(), trees_dead

def summer():
    global trees_dead
    global foods
    global board

    if trees_dead == []:
        #print(trees, board, foods)
        return
    for tree in trees_dead:
        x, y, food = tree[0], tree[1], tree[2]
        board[x][y] += food
    #print(trees, board, foods)
    return

def fall():
    global n
    global trees
    global trees_remain
    tmp = []
    for tree in trees_remain:
        x, y, age = tree[0], tree[1], tree[2]
        for dx, dy in reproduce:
            nx, ny = x + dx, y + dy
            if nx < 0: continue
            if n-1 < nx: continue
            if ny < 0: continue
            if n-1 < ny: continue
            if age % 5 == 0:
                tmp.append((nx,ny, 1))
    trees = tmp + trees
                #trees.insert(0, (nx, ny, 1))
    #print(trees, board, foods)
    return

def winter():
    global board
    global foods

    for i in range(n):
        for j in range(n):
            board[i][j] += foods[i][j]
    #print(trees, board, foods)
    return

for i in range(k):
    trees, trees_dead = spring()
    summer()
    fall()
    winter()
    if len(trees) == 0:
        print(0)
        break
    if i == k-1:
        print(len(trees))
~~~


이 문제의 경우 insert를 사용하면 시간초과가 일어납니다. insert의 시간 복잡도는 O(n)으로 시간이 꽤 오래 걸립니다.  
참고로 copy()의 경우에도 O(n)의 시간복잡도를 가지고 있으므로 copy()를 추가로 사용한다면 시간초과가 일어날 수 있습니다.   






궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
