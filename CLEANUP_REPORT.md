# Cleanup Report

생성일: 2026-06-09

## 1. 작업 목표

현재 GitHub 연동 저장소 안의 `ai` 폴더를 새 저장소로 분리하지 않고, 기존 Git 연결과 커밋 기록을 유지한 채 용도별 구조로 재분류했습니다.

## 2. 정리 전 주요 구조

```text
AI_PROJECTS/
AI_Study/
Country_Economy_Project/
ECONOMY_PROJECT/
World_Happiness/
Vibe_project/
vibe_coding/
short_code/
docs/
projects/
experiments/
programming/
tools/
reports/
archive/
```

## 3. 정리 후 주요 구조

```text
projects/
├─ country-economy-project/
├─ economy-project/
└─ world-happiness/
tools/
├─ repo_cleaner/
experiments/
├─ ai-projects/
├─ short-code/
├─ vibe-coding/
└─ vibe-project/
programming/
coursework/
└─ AI_Study/
prompt-decks/
docs/
reports/
archive/
└─ cleanup-candidates/
```

## 4. 이동한 파일과 폴더

| 이전 위치 | 새 위치 | 이유 |
| --- | --- | --- |
| `AI_Study/` | `coursework/AI_Study/` | 수업/실습 원본 자료로 보존 |
| `Country_Economy_Project/` | `projects/country-economy-project/` | 독립 데이터 분석 프로젝트 |
| `ECONOMY_PROJECT/` | `projects/economy-project/` | 독립 데이터 분석 프로젝트 |
| `World_Happiness/` | `projects/world-happiness/` | 독립 데이터 분석 프로젝트 |
| `AI_PROJECTS/` | `experiments/ai-projects/` | 짧은 선형회귀 실험 |
| `short_code/` | `experiments/short-code/` | 짧은 코딩 실험 |
| `vibe_coding/` | `experiments/vibe-coding/` | 단일 노트북 실험 |
| `Vibe_project/` | `experiments/vibe-project/` | Jupyter 기반 소형 실험 |
| `README.md.md` | `archive/cleanup-candidates/README.md.md` | 인코딩이 깨져 보이는 이전 루트 문서 보관 |

## 5. 이름을 변경한 항목

- `Country_Economy_Project` -> `country-economy-project`
- `ECONOMY_PROJECT` -> `economy-project`
- `World_Happiness` -> `world-happiness`
- `AI_PROJECTS` -> `ai-projects`
- `short_code` -> `short-code`
- `vibe_coding` -> `vibe-coding`
- `Vibe_project` -> `vibe-project`

## 6. 코드에서 수정한 경로

코드 import나 실행 파일의 내부 상대 경로는 수정하지 않았습니다. 이동은 프로젝트/폴더 단위로 수행했기 때문에 각 폴더 내부 상대 경로는 유지됩니다.

단, 외부 문서의 예전 경로는 새 구조에 맞춰 갱신했습니다.

## 7. 중복/삭제 후보

삭제하지 않고 후보로만 기록합니다.

- `coursework/AI_Study/**/__pycache__/`
- `coursework/AI_Study/**/*.pyc`
- `**/.ipynb_checkpoints/`
- `coursework/AI_Study/**/venv/`
- `coursework/AI_Study/**/staticfiles/`
- `coursework/AI_Study/**/_staticfiles/`
- `coursework/AI_Study/08_LLM/chroma*/`
- `archive/cleanup-candidates/README.md.md`

파일명 기준 중복 후보는 `__init__.py`, `README.md`, Django/Flask 공통 파일명, RAGAS 결과 파일명 등 다수입니다. 대부분 프로젝트 내부 구조상 자연스러운 중복일 수 있어 삭제하지 않았습니다.

## 8. 민감정보 후보

실제 값은 출력하지 않습니다.

- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/.env`
- `coursework/AI_Study/**/settings.py`
- `coursework/AI_Study/08_LLM/**/*.ipynb`
- `coursework/AI_Study/**/rag_module.py`
- `experiments/vibe-project/Secret_Key*.ipynb`
- `experiments/vibe-project/password_tool*.ipynb`

## 9. 대용량 후보

- `coursework/AI_Study/프로젝트 1/data/전처리/*.csv`
- `coursework/AI_Study/프로젝트 1/data/훈련데이터셋/*.csv`
- `coursework/AI_Study/chatbot_app/kor.traineddata`
- `coursework/AI_Study/chatbot_app1/kor.traineddata`
- `coursework/AI_Study/**/*.h5`
- `coursework/AI_Study/**/*.pkl`
- `coursework/AI_Study/**/*.joblib`

## 10. 실행/테스트 결과

대부분 노트북/수업 자료라 전체 실행 테스트는 수행하지 않았습니다. 구조 분석 도구는 정리 후 다시 실행해 보고서를 갱신할 예정입니다.

## 11. 보류한 작업

- 파일 삭제
- Git LFS 적용
- Git 추적 해제
- 민감정보 파일 내용 수정
- 수업 원본 내부 구조 재배치
- `coursework/AI_Study/프로젝트 1`, `프로젝트 2`의 독립 프로젝트 승격

