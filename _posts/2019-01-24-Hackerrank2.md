---
layout: post
title:  "Hackerrank 레벨업 하기42 [Library Fine]"
date:   2019-01-24
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

Hackerrank 레벨업하기  
문제   
[Library Fine](https://www.hackerrank.com/challenges/library-fine/problem)

~~~
def libraryFine(d1, m1, y1, d2, m2, y2):
    if y1 > y2:
        return 10000
    else:
        if y1 < y2:
            return 0
        else:
            if m1 > m2:
                return 500 * (m1 -m2)
            else:
                if m1 < m2:
                    return 0
                else:
                    if d1 > d2:
                        return 15 * (d1 - d2)
                    else:
                        return 0
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
