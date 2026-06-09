# Project Index

현재 저장소의 주요 폴더와 분류 기준입니다. 기존 파일은 삭제하지 않고, 프로젝트 단위로만 재배치했습니다.

## 최상위 분류

| 위치 | 분류 | 설명 |
| --- | --- | --- |
| `projects/country-economy-project/` | 데이터 분석 | 국가별 GDP, 수출, 군사비, 특허, 인구 등 경제 지표 분석 |
| `projects/economy-project/` | 데이터 분석 | 환율/경제 지표 분석 및 예측 실험 |
| `projects/world-happiness/` | 데이터 분석 | 세계 행복 지수 분석 |
| `coursework/AI_Study/` | 수업/실습 원본 | Python, 웹, DB, ML/DL, NLP, LLM, Flask, Django, 과제 자료 |
| `experiments/ai-projects/` | 실험 | 선형 회귀 노트북 중심의 짧은 실험 |
| `experiments/short-code/` | 실험 | 짧은 코딩 연습, Jupyter 테트리스 |
| `experiments/vibe-coding/` | 실험 | 계산 관련 단일 노트북 |
| `experiments/vibe-project/` | 실험 | 숫자 맞히기, 비밀번호 도구 등 Jupyter 제작물 |
| `programming/` | 일반 프로그래밍 학습 | Python, 웹, 백엔드, DB, 네트워크, 보안, DevOps, 테스트 |
| `tools/repo_cleaner/` | 도구 | 저장소 구조/위험 후보 분석기 |
| `docs/` | 문서 | 인덱스, 로드맵, 작업 지침 |
| `reports/` | 보고서 | repo cleaner 결과와 정리 보고서 |
| `archive/cleanup-candidates/` | 보관 후보 | 삭제하지 않고 보존한 이전/중복 후보 |

## 정리 후보

삭제하지 않고 후보로만 기록합니다.

| 후보 | 위치 | 이유 |
| --- | --- | --- |
| 기존 깨진 README | `archive/cleanup-candidates/README.md.md` | 인코딩이 깨져 보이는 이전 루트 문서 |
| Python 캐시 | `coursework/AI_Study/**/__pycache__/`, `*.pyc` | 실행 캐시 |
| Jupyter 체크포인트 | `**/.ipynb_checkpoints/` | 자동 생성 파일 |
| 가상환경 | `coursework/AI_Study/**/venv/` | 재생성 가능한 로컬 환경 |
| Chroma DB | `coursework/AI_Study/08_LLM/chroma*/` | 로컬 벡터 DB 산출물 |
| Django staticfiles | `coursework/AI_Study/**/staticfiles/`, `_staticfiles/` | collectstatic 산출물 |
| 대형 CSV/모델 | `coursework/AI_Study/프로젝트 1/data/`, `*.h5`, `*.pkl`, `*.joblib` | Git LFS 또는 외부 보관 검토 |

## 민감정보 후보

실제 값은 출력하지 않습니다.

| 위치 | 위험 유형 |
| --- | --- |
| `coursework/AI_Study/lease_law_app-main/lease_law_app-main/.env` | 실제 환경변수 파일 |
| `coursework/AI_Study/**/settings.py` | Django secret, DB 설정 후보 |
| `coursework/AI_Study/08_LLM/**/*.ipynb` | OpenAI/Pinecone/Upstage API 키 참조 후보 |
| `experiments/vibe-project/Secret_Key*.ipynb` | 파일명 기준 secret/key 후보 |
| `experiments/vibe-project/password_tool*.ipynb` | password 후보 |

