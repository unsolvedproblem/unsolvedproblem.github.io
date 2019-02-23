---
layout: post
title:  "Hackerrank 30 Days of Code 18 [Queues and Stacks]"
date:   2019-02-23
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Queues and Stacks](https://www.hackerrank.com/challenges/30-queues-stacks/problem)

~~~
class Solution:
    stack = []
    queue = []
    def pushCharacter(self, cha):
        self.stack.append(cha)

    def enqueueCharacter(self, cha):
        self.queue.append(cha)

    def popCharacter(self):
        return self.stack.pop()

    def dequeueCharacter(self):
        return self.queue.pop(0)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
