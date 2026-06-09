# Programming Study Lab

이 폴더는 Python, JavaScript, TypeScript, HTML/CSS, 백엔드, 프론트엔드, 데이터베이스, 네트워크, 보안, DevOps, 테스트, 자동화, CLI 도구 등 인터넷 프로그램 개발에 필요한 일반 기술을 공부하기 위한 공간이다.

## 목적

`programming/`은 AI 모델 자체보다 프로그램을 만들고 운영하는 기본기를 쌓는 공간이다. 작은 실습, 미니 프로젝트, 웹앱 구조 연습, API 서버, 데이터베이스, 배포와 테스트를 단계적으로 정리한다.

## AI 프로젝트와 분리하는 이유

- `projects/`는 AI / ML / LLM 중심의 완성형 프로젝트를 둔다.
- `coursework/AI_Study/`는 기존 수업/실습 원본 자료로 유지한다.
- `programming/`은 AI 기능을 붙이기 전 필요한 일반 개발 역량을 따로 연습한다.
- 이렇게 나누면 AI 실험 코드와 일반 웹/백엔드/프로그래밍 학습 코드가 섞이지 않아 찾기 쉽다.

## 하위 폴더

| 폴더 | 설명 |
| --- | --- |
| `01_python/` | Python 문법, 파일 처리, 자동화 기초 |
| `02_javascript/` | JavaScript 문법, DOM, 비동기 처리 |
| `03_typescript/` | TypeScript 타입 시스템과 구조화 |
| `04_html_css/` | HTML 구조, CSS 레이아웃, 반응형 UI |
| `05_web_frontend/` | 프론트엔드 앱 구조와 상태 관리 |
| `06_backend/` | HTTP, REST API, Flask/FastAPI/Django |
| `07_database/` | SQL, SQLite, PostgreSQL, ORM |
| `08_network/` | HTTP, DNS, TCP/UDP, client/server |
| `09_security_basics/` | 인증, 세션, 쿠키, JWT, 웹 보안 기초 |
| `10_devops/` | Docker, CI/CD, 배포, 로깅과 모니터링 |
| `11_cli_tools/` | 명령줄 도구와 개발 생산성 |
| `12_automation/` | 반복 작업 자동화와 스크립트 |
| `13_algorithms/` | 자료구조, 알고리즘, 문제 해결 |
| `14_design_patterns/` | 설계 패턴과 코드 구조 |
| `15_testing/` | 테스트, 디버깅, 품질 관리 |
| `mini_projects/` | 여러 주제를 묶은 작은 완성형 실습 |

## 추천 학습 순서

1. `01_python/`
2. `04_html_css/`
3. `02_javascript/`
4. `08_network/`
5. `06_backend/`
6. `07_database/`
7. `05_web_frontend/`
8. `09_security_basics/`
9. `15_testing/`
10. `10_devops/`
11. `12_automation/`
12. `11_cli_tools/`
13. `03_typescript/`
14. `13_algorithms/`
15. `14_design_patterns/`

## 프로젝트 규칙

- 새 실습은 해당 주제 폴더 아래에 작은 폴더로 만든다.
- 완성형 미니 프로젝트는 `mini_projects/` 또는 주제별 폴더 안에 둔다.
- 각 프로젝트에는 최소한 `README.md`를 만든다.
- 실행 방법, 배운 개념, 다음 개선 아이디어를 README에 기록한다.
- AI 모델/API 중심 프로젝트로 커지면 `projects/`로 분리할지 검토한다.

## Git에 올리지 않는 것이 좋은 파일

- `.env`, `.env.*`
- API key, secret, password, token, credential이 들어간 파일
- `venv/`, `.venv/`, `node_modules/`
- `__pycache__/`, `*.pyc`
- `.ipynb_checkpoints/`
- SQLite DB, Chroma DB, 로컬 캐시
- 빌드 산출물, 로그, 임시 파일
- 대용량 데이터, 모델 파일, 실행 파일

## 첫 프로젝트 추천

첫 일반 프로그래밍 프로젝트는 `06_backend/rest_api_practice/`에서 시작한다. FastAPI 또는 Flask로 간단한 REST API 서버를 만들며 HTTP, routing, JSON, status code, SQLite, CRUD를 한 번에 연습한다.
