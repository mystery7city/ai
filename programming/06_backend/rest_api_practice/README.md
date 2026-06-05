# REST API Practice

## 목표

FastAPI 또는 Flask로 간단한 REST API 서버를 만들면서 인터넷 프로그램의 기본 구조를 배운다.

## 1차 구현 기능

- GET `/items`
- GET `/items/{id}`
- POST `/items`
- PUT `/items/{id}`
- DELETE `/items/{id}`
- SQLite 저장
- JSON request/response
- status code 사용
- 간단한 README와 실행 방법

## 배울 개념

- HTTP method
- REST API
- routing
- request body
- response body
- status code
- SQLite
- CRUD
- API 문서

## 추천 파일 구조

```text
rest_api_practice/
├─ README.md
├─ requirements.txt
├─ app.py
├─ database.py
├─ models.py
├─ schemas.py
├─ crud.py
├─ data/
│  └─ .gitkeep
└─ tests/
   └─ .gitkeep
```

## 이번 단계에서 하지 않는 것

- 실제 서버 코드 구현
- 패키지 설치
- 외부 API 연동
- `.gitignore` 수정

## 완료 기준

다음 단계에서 CRUD API가 동작하고, SQLite에 데이터가 저장되며, README에 실행 방법과 API 예시가 정리되어 있으면 완료로 본다.

## 다음 단계

FastAPI 또는 Flask 중 하나를 선택해 실제 구현을 시작한다.

