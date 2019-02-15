---
layout: post
title:  "Hackerrank 30 Days of Code 15 [Linked List]"
date:   2019-02-15
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Linked List](https://www.hackerrank.com/challenges/30-linked-list/problem)

~~~
def insert(self,head,data):
    new_node = Node(data)
    if head == None: return new_node
    P_node = head
    T_node = P_node.next
    while T_node != None:
        P_node = T_node
        T_node = P_node.next
    P_node.next = new_node
    return head
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
