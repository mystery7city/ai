# 01. Hello C++

## 이번 예제에서 배우는 개념

- `#include`를 사용해 표준 라이브러리 불러오기
- `main` 함수와 프로그램의 시작점 이해하기
- `std::cout`으로 화면에 문자열 출력하기
- `std::getline`과 `std::cin`으로 사용자 입력받기
- `std::string`에 문자열 저장하기
- `return 0`으로 프로그램이 정상 종료되었음을 알리기

## 컴파일 명령어

예제 폴더에서 다음 명령어를 실행합니다.

```bash
g++ -std=c++17 -Wall -Wextra -pedantic main.cpp -o hello_cpp
```

Windows에서 Microsoft C++ 컴파일러를 사용한다면 다음 명령어를 사용할 수 있습니다.

```powershell
cl /std:c++17 /EHsc main.cpp /Fe:hello_cpp.exe
```

## 실행 방법

Linux 또는 macOS:

```bash
./hello_cpp
```

Windows PowerShell:

```powershell
.\hello_cpp.exe
```

프로그램이 시작되면 이름을 입력하고 Enter 키를 누릅니다.

## 연습 문제

1. 환영 메시지에 사용자의 이름뿐 아니라 현재 배우고 싶은 C++ 주제도 함께 출력해 보세요.
2. 사용자의 나이를 추가로 입력받아 내년에 몇 살이 되는지 출력해 보세요.
3. 시작 메시지와 환영 메시지를 각각 별도의 함수로 분리해 보세요.
