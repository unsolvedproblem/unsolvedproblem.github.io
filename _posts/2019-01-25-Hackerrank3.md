---
layout: post
title:  "Hackerrank 레벨업 하기48 [Queen's Attack 2]"
date:   2019-01-25
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Queen's Attack 2](https://www.hackerrank.com/challenges/queens-attack-2/problem)

~~~
def queensAttack(n, k, r_q, c_q, obstacles):
    x, y = r_q, c_q
    o = obstacles
    actual_o1 = []
    actual_o2 = []
    actual_o3 = []
    actual_o4 = []

    for i in o:
        if i[0] == x:
            actual_o1.append(i)
        elif i[1] == y:
            actual_o2.append(i)  
        elif (i[0] - x) == (i[1] - y):
            actual_o3.append(i)
        elif (i[0] - x) == -(i[1] - y):
            actual_o4.append(i)

    for i in [actual_o1, actual_o2, actual_o3, actual_o4]:
        i.sort()

    result = 0

    if actual_o1 == []:
        result += n - 1
    else:
        if y < actual_o1[0][1]:
            result += actual_o1[0][1] - 2
        elif actual_o1[-1][1] < y:
            result += n - actual_o1[-1][1] - 1
        else:
            for i in range(len(actual_o1) - 1):
                if actual_o1[i][1] < y < actual_o1[i+1][1]:
                    result += actual_o1[i+1][1] - actual_o1[i][1] - 2
                    break

    if actual_o2 == []:
        result += n - 1
    else:
        if x < actual_o2[0][0]:
            result += actual_o2[0][0] - 2
            print(result)
        elif actual_o2[-1][0] < x:
            result += n - actual_o2[-1][0] - 1
        else:
            for i in range(len(actual_o2) - 1):
                if actual_o2[i][0] < x < actual_o2[i+1][0]:
                    result += actual_o2[i+1][0] - actual_o2[i][0] - 2

    if actual_o3 == []:
        result += min(x-1,y-1) + min(n - x, n - y)
    else:
        if x < actual_o3[0][0]:
            result += min(actual_o3[0][0] -2,actual_o3[0][1] -2)
        elif x > actual_o3[-1][0]:
            result += min(n - actual_o3[-1][0] -1,n - actual_o3[-1][1] -1)
        else:
            for i in range(len(actual_o3) - 1):
                if actual_o3[i][0] < x < actual_o3[i + 1][0]:
                    result += actual_o3[i+1][0] - actual_o3[i][0] - 2

    if actual_o4 == []:
        result += min(n-x, y - 1) + min(x -1, n - y)
    else:
        if x < actual_o4[0][0]:
            result += min(actual_o4[0][0]-2, n- actual_o4[0][1] - 1)
        elif x > actual_o4[-1][0]:
            result += min(actual_o4[-1][1]-2, n-actual_o4[-1][0] - 1)
        else:
            for i in range(len(actual_o4)-1):
                if actual_o4[i][0] < x < actual_o4[i + 1][0]:
                    result += actual_o4[i+1][0] - actual_o4[i][0] - 2
    return result
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
