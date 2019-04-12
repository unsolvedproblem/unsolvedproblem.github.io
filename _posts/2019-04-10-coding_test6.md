---
layout: post
title:  "드래곤 커브 삼성 coding test preparing6 python3 파이썬3"
date:   2019-04-10
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[드래곤 커브](https://www.acmicpc.net/problem/15685)

~~~
n = int(input())
information = []
for i in range(n):
    tmp = list(map(int, input().split()))
    information.append(tmp)

def mat_mul(mat, vec):
    return [mat[0][0] * vec[0] + mat[0][1] * vec[1], mat[1][0] * vec[0] + mat[1][1] * vec[1]]
def mat_add(vec1, vec2):
    return [vec1[0]+ vec2[0], vec1[1] + vec2[1]]
def rot90(arr):
    rotation = []
    tmp = [[0, -1],[1, 0]]
    for data in arr:
        x, y = data[0], data[1]
        rotation.append(mat_mul(tmp, [x, y]))
    dx, dy = arr[-1][0] - rotation[-1][0], arr[-1][1] - rotation[-1][1]
    final_rot = []
    for vec in rotation[:-1]:
        final_rot = [mat_add(vec, [dx, dy])] + final_rot
    return final_rot
board = []
for i in information:
    vec = []
    x, y, direction, generation = i
    vec.append([x,y])
    if direction == 0: vec.append([x + 1, y])
    elif direction == 1: vec.append([x, y - 1])
    elif direction == 2: vec.append([x - 1, y])
    elif direction == 3: vec.append([x, y + 1])
    for j in range(generation):
        vec = vec + rot90(vec)
    board = board + vec
del vec
max_board = 0
for i in board:
    max_tmp = max(i)
    if max_tmp > max_board: max_board = max_tmp
count = 0
for i in range(max_board):
    for j in range(max_board):
        if [i, j] in board and [i+1, j] in board and [i, j+1] in board and [i+1, j+1] in board:
            count += 1
print(count)
~~~
혹시 itertools를 사용하지 못한다면 직접 구현하자.
~~~
visit = []
def combination(arr, m):
    if len(visit) == m:
        yield visit
    else:
        for j in range(len(arr)):
            visit.append(arr[j])
            yield from combination(arr[j+1:], m)
            visit.pop()

for i in combination([1,2,3,4,5], 3):
    print(i)

def permutation(arr, i=0):
    if len(arr) - 1 == i:
        yield arr
        return
    else:
        for j in range(i, len(arr)):
            arr[i],arr[j] = arr[j], arr[i]
            yield from permutation(arr, i+1)
            arr[i],arr[j] = arr[j], arr[i]
permutation([1,2,3,4])
~~~








궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
