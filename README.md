# AI Personal Research Lab

이 저장소는 AI / 프로그래밍 / 인터넷 기술을 공부하고 실험하는 개인 개발 연구소입니다. 기존 GitHub 저장소와 커밋 기록을 유지하면서, 자료의 성격에 따라 폴더를 나누어 관리합니다.

## 폴더 구조

```text
ai/
├─ projects/         # 독립적으로 실행 가능한 AI, ML, LLM 프로젝트
├─ tools/            # 실제 작업에 사용하는 유틸리티와 자동화 도구
├─ experiments/      # API, 모델, 라이브러리, 짧은 기능 실험
├─ programming/      # Python, 웹, 알고리즘 등 일반 프로그래밍 학습
├─ coursework/       # 학교 수업, 과제, 강의 실습 원본
│  └─ AI_Study/
├─ prompt-decks/     # 이미지/텍스트 생성 프롬프트와 프롬프트 덱
├─ docs/             # 개념 정리, 사용법, 기술 문서
├─ reports/          # 결과 보고서, 분석 결과, 발표 자료
└─ archive/          # 현재 사용하지 않지만 삭제하기 애매한 자료
```

## 각 폴더의 역할

| 폴더 | 역할 |
| --- | --- |
| `projects/` | 완성형 또는 독립 실행이 가능한 AI/ML/LLM 프로젝트 |
| `tools/` | 저장소 분석기, 자동화 스크립트, 개발 보조 도구 |
| `experiments/` | 짧은 노트북, API 테스트, 아이디어 검증 코드 |
| `programming/` | Python, JavaScript, 백엔드, DB, 네트워크, 보안, DevOps 학습 |
| `coursework/` | 수업/강의/과제 원본 자료. 원본 보존을 우선합니다. |
| `prompt-decks/` | 프롬프트 묶음과 생성형 AI 입력 템플릿 |
| `docs/` | 저장소 운영 문서, 인덱스, 로드맵 |
| `reports/` | 자동 분석 결과와 정리 보고서 |
| `archive/` | 삭제하기 애매한 이전 문서, 정리 후보, 보관 자료 |

## 주요 프로젝트

| 위치 | 설명 |
| --- | --- |
| `projects/country-economy-project/` | 국가 경제 지표 데이터 분석 프로젝트 |
| `projects/economy-project/` | 환율/경제 지표 분석 프로젝트 |
| `projects/world-happiness/` | 세계 행복 지수 분석 프로젝트 |
| `tools/repo_cleaner/` | 저장소 구조, 민감정보 후보, Git 제외 후보를 점검하는 읽기 전용 분석 도구 |
| `programming/06_backend/rest_api_practice/` | REST API 학습 프로젝트 계획 문서 |

## 실행 방법

프로젝트마다 실행 방식이 다릅니다. 각 프로젝트 폴더의 `README.md` 또는 노트북을 먼저 확인합니다.

저장소 분석 도구는 다음처럼 실행합니다.

```bash
py tools/repo_cleaner/repo_cleaner.py --large-threshold-mb 10
```

보고서는 아래에 생성됩니다.

```text
reports/repo_cleaner_report.md
reports/repo_cleaner_report.json
```

## 새 파일을 넣는 규칙

- 독립 프로젝트: `projects/project-name/`
- 짧은 실험: `experiments/experiment-name/`
- 일반 프로그래밍 학습: `programming/topic-name/`
- 수업 원본: `coursework/`
- 자동화 도구: `tools/tool-name/`
- 프롬프트 자료: `prompt-decks/`
- 문서: `docs/`
- 분석/결과 보고서: `reports/`
- 삭제하기 애매한 이전 자료: `archive/cleanup-candidates/`

## Git에 올리지 않는 것이 좋은 파일

- `.env`, `.env.*`
- API key, token, secret, password, credential이 들어간 파일
- `venv/`, `.venv/`, `node_modules/`
- `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
- SQLite DB, Chroma DB, 로컬 캐시
- 빌드 결과물, 로그, 임시 파일
- 대형 모델 파일과 로컬 데이터셋

중요한 대형 데이터나 모델이 이미 프로젝트 실행에 필요하다면 삭제하지 말고 Git LFS 또는 외부 보관 여부를 먼저 결정합니다.

