---
layout: post
title:  "Ubuntu Matplotlib 한글 인식 방법"
date:   2020-08-04
category: etc
tags: ubuntu, matplotlib, korean
author: Diominor, 백승열
comments: true
---

참조 : [http://taewan.kim/post/matplotlib_hangul](http://taewan.kim/post/matplotlib_hangul/)

방법 중에서도 anaconda가 설치 되어 있을 때

1. 우분투 한글 폰트 설치
    ```
    $ sudo apt-get install -y fonts-nanum fonts-nanum-coding fonts-nanum-extra
    ```
    재부팅 하면 폰트가 인식 되지만 재부팅 하기 싫을 경우 아래 코드 실행
    ```
    $ fc-cache -fv
    ```
    
2. 폰트 anaconda의 matplotlib에 옮기기

   시스템에 설치된 나눔 글꼴을 아나콘다의 matplotlib패키지 안의 fonts/ttf 파일로 옮겨줍니다.
   ```
   $ sudo cp /usr/share/fonts/truetype/nanum/Nanum* ~/anaconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/
   ```
   (anaconda, python의 버전 및 개인 설정에 따라 '.../fonts/ttf/'의 경로는 바뀔 수 있으니 확인하고 입력해주세요.)
    
2. Matplotlib 폰트 정보 확인

    matplotlib은 사용할 font 정보를 fontlist-v310.json에 관리합니다.
    
    (이름이 'fontlist-v310.json'와 정확히 일치하지 않을 수도 있습니다.)
    ```
    $ ls -al ~/.cache/matplotlib/
    합계 104
    drwxr-xr-x  3 backgom2357 backgom2357  4096  6월 17 08:05 .
    drwx------ 27 backgom2357 backgom2357  4096  6월 16 13:59 ..
    -rw-r--r--  1 backgom2357 backgom2357 92708  6월 17 08:05 fontlist-v310.json
    drwxr-xr-x  2 backgom2357 backgom2357  4096  6월  2 11:44 tex.cache
    ```

3. json 수정

    vim을 이용해 json을 열어 font 정보를 추가합니다.
    ```
    $ vim ~/.cache/matplotlib/fontlist-v310.json
    ```
    
    - fontlist-v310.json
       
       "ttflist": \[...\] 폰트 정보를 양식에 따라 리스트에 추가합니다.
    ```
    {
      "_version": 310,
      "_FontManager__default_weight": "normal",
      "default_size": null,
      "defaultFamily": {
        "ttf": "DejaVu Sans",
        "afm": "Helvetica"
      },
      "ttflist": [
      
    { 
      "fname": "fonts/ttf/[폰트 이름].ttf",
      "name": "[폰트 이름]",
      "style": "normal",
      "variant": "normal",
      "weight": 400,
      "stretch": "normal",
      "size": "scalable",
      "__class__": "FontEntry"

    },
    {
      "fname": "fonts/ttf/STIXGeneralItalic.ttf",
      "name": "STIXGeneral",
      "style": "normal",
      "variant": "normal",
      "weight": 400,
      "stretch": "normal",
      "size": "scalable",
      "__class__": "FontEntry"
    },
    ...
    ]
    ...

    ```
    (\[폰트이름\] 예 : NanumGothic)
    

4. 확인하기
    
    console, jupyter notebook 등을 통해 확인합니다.
    ```
    import matplotlib
    import matplotlib.font_manager as fm
    font_list = fm.fontManager.ttflist
    [f.name for f in font_list if 'Nanum' in f.name]
    # ["[폰트 이름]"]
    ```
    
5. 사용하기

    matplotlib에 원하는 폰트를 설정하는 코드는 여러 가지가 있지만 그 중 하나를 소개합니다.
    ```
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font', family="[폰트 이름]")