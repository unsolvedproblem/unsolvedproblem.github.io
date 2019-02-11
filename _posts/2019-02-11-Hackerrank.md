---
layout: post
title:  "Hackerrank 30 Days of Code 12 [Inheritance]"
date:   2019-02-11
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

기초를 갈고 닦기 위한 스탭!  
Hackerrank 30 Days of Code  
문제   
[Inheritance](https://www.hackerrank.com/challenges/30-inheritance/problem)

~~~
class Student(Person):
    #   Class Constructor
    #   
    #   Parameters:
    #   firstName - A string denoting the Person's first name.
    #   lastName - A string denoting the Person's last name.
    #   id - An integer denoting the Person's ID number.
    #   scores - An array of integers denoting the Person's test scores.
    #
    # Write your constructor here
    def __init__(self, firstName, lastName, idNumber, scores):
        self.firstName = firstName
        self.lastName = lastName
        self.idNumber = idNumber
        self.scores = sum(scores)/ len(scores)
    #   Function Name: calculate
    #   Return: A character denoting the grade.
    #
    # Write your function here
    def calculate(self):
        if 90<= self.scores <= 100: return 'O'
        elif 80 <= self.scores < 90: return 'E'
        elif 70 <= self.scores < 80: return 'A'
        elif 55 <= self.scores < 70: return 'P'
        elif 40 <= self.scores < 55: return 'D'
        else:
            return "T"
~~~

궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com  
