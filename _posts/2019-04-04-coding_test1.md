---
layout: post
title:  "보물상자 비밀번호 삼성 coding test preparing1 python3 파이썬3"
date:   2019-04-04
category: code_practice
tags: coding_practice
author: Khel Kim, 김현호
comments: true
---

코딩 테스트 준비  

문제   
[보물상자 비밀번호](https://www.swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AWXRUN9KfZ8DFAUo)

~~~
T = int(input())

def get_num(N, K, string):
    global n
    n = int(N / 4)
    num = []
    for i in range(1, n+1):
        tmp_str = string[-i:] + string[:-i]
        for j in range(4):
            if '0x' + tmp_str[j * n: (j + 1) * n] not in num:
                num.append('0x' + tmp_str[j * n:(j + 1) * n])
    return num

# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T + 1):
    N, K = map(int, input().split())
    string = input()
    num_16 = get_num(N, K, string)
    num = []
    for i in num_16:
        num.append(int(i, 16))
    num.sort(reverse=True)
    print('#%s' %test_case, num[K-1])
~~~

- 2진수: 0b
- 8진수: 0o
- 16진수: 0x

~~~
print(bin(42), oct(42), hex(42)
~~~
~~~
##결과
'0b101010' '0o52' '0x2a'
~~~
~~~
int('0b101010', 2)
int('0o52', 8)
int('0x2a', 16)
~~~
~~~
##결과
42
42
42
~~~









궁금하신게 있다면 메일 주세요.  
k2h7913@daum.net  
cafehero123@gmail.com    
