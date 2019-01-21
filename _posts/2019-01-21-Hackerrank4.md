---
layout: post
title:  "Hackerank 레벨업 하기29"
date:   2019-01-21
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
---

Hackerrank 레벨업하기  
문제  
[Designer PDF Viewer](https://www.hackerrank.com/challenges/designer-pdf-viewer/problem)

~~~
def designerPdfViewer(h, word):
    alpha = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    count = len(word)
    word = list(set(list(word)))
    word_index = []
    heights = []
    for i in word:
        word_index.append(alpha.index(i))
    for i in word_index:
        heights.append(h[i])
    return count * max(heights)
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com
