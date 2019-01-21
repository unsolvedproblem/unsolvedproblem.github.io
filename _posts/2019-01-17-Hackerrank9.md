---
layout: post
title:  "Hackerank 레벨업 하기10 [Grading Students]"
date:   2019-01-18
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제  
[Grading Students](https://www.hackerrank.com/challenges/grading/problem)

~~~
def gradingStudents(grades):
    section=[]
    for i in range(8, 21):
        section.append(5*i)
    for i in range(len(grades)):
        if grades[i] <= 37:
            pass
        else:
            tmp = list(filter(lambda x: x>0, map(lambda x: x-grades[i], section)))
            if tmp == []:
                pass
            elif tmp[0]<3:
                grades[i] = grades[i] + tmp[0]
            else:
                pass
    return grades
~~~

궁금하신게 있다면 메일 주세요.   
k2h7913@daum.net  
cafehero123@gmail.com
