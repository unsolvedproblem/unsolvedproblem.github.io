---
layout: post
title:  "Hackerrank 30 Days of Code 23 [BST Level-Order Traversal]"
date:   2019-03-16
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[BST Level-Order Traversal](https://www.hackerrank.com/challenges/30-binary-trees/problem)

~~~
def levelOrder(self,root):
        #Write your code here
      levelq = []
      levelq.append(root)
      if root != None:
          while levelq != []:
              tmp = levelq.pop(0)
              print(tmp.data, end=' ')

              if tmp.left != None:
                  levelq.append(tmp.left)
              if tmp.right != None:
                  levelq.append(tmp.right)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
