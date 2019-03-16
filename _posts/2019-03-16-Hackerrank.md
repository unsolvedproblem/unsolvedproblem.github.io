---
layout: post
title:  "Hackerrank 30 Days of Code 22 [Binary Search Trees]"
date:   2019-03-16
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Binary Search Trees](https://www.hackerrank.com/challenges/30-binary-search-trees/problem)

~~~
def getHeight(self,root):
    #Write your code here
    if root == None:
        return -1
    else:
        return 1 + max(self.getHeight(root.left),
                    self.getHeight(root.right))
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
