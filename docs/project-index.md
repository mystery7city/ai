# Project Index

이 문서는 `ai` 폴더 안의 기존 파일과 하위 폴더를 분석해 분류한 초안입니다.
이번 작업에서는 기존 파일을 삭제하거나 이동하지 않았습니다.

## 최상위 구조

| 경로 | 파일 수 | 현재 성격 | 권장 분류 |
| --- | ---: | --- | --- |
| `AI_PROJECTS/` | 2 | 선형 회귀 노트북 중심의 짧은 프로젝트 | `experiments/` 또는 `projects/linear-regression-practice/` |
| `AI_Study/` | 5316 | Python, 웹, DB, 딥러닝, NLP, LLM, Django/Flask, 팀 프로젝트 자료가 섞인 대형 학습 폴더 | `archive/ai-study-original/` 보관 후 선별 분리 제안 |
| `Country_Economy_Project/` | 14 | 국가 경제 지표 데이터 분석 노트북과 CSV | `projects/country-economy-analysis/` |
| `ECONOMY_PROJECT/` | 15 | 환율, 경제 지표, 예측 데이터가 있는 분석 프로젝트 | `projects/exchange-rate-analysis/` |
| `World_Happiness/` | 4 | 세계 행복 지수 분석 프로젝트 | `projects/world-happiness-analysis/` |
| `Vibe_project/` | 5 | 숫자 맞히기, 비밀번호 도구 등 Jupyter 기반 소형 제작물 | `experiments/vibe-project/` |
| `vibe_coding/` | 1 | 계산 관련 Jupyter 노트북 | `experiments/vibe-coding/` |
| `short_code/` | 3 | 짧은 코딩 연습, 테트리스 노트북 | `experiments/short-code/` |

## 파일 유형 요약

| 유형 | 개수 | 메모 |
| --- | ---: | --- |
| `.py` | 1885 | Django/Flask 앱, 모듈, 학습 코드가 다수 포함됨 |
| `.pyc` | 1779 | 실행 캐시 파일로 보임. 추후 정리 후보 |
| `.csv` | 278 | 분석용 데이터 파일 다수 |
| 확장자 없음 | 214 | 실행 파일, 설정, 기타 자료 가능성 |
| `.html` | 191 | 웹/Django/Flask 관련 템플릿 및 정적 결과 가능성 |
| `.js` | 185 | 웹 학습/앱 자료 |
| `.ipynb` | 138 | 분석 및 실험 노트북 |
| `.jsonl` | 118 | RAG/평가/데이터셋 후보 |
| `.json` | 97 | 설정, 결과, 모델 메타데이터 |
| `.txt` | 82 | 텍스트 자료 및 로그 후보 |
| `.png`, `.jpg`, `.svg` | 122 | 결과 이미지, 스크린샷, 정적 자산 |
| `.md` | 9 | 기존 문서 |

## 현재 눈에 띄는 정리 포인트

- 루트에 `README.md.md`가 있고 내용 인코딩이 깨져 보입니다. 삭제하지 않고 새 `README.md`를 추가했습니다.
- `AI_Study/`는 학습 자료, 웹 프레임워크 실습, RAG/챗봇 앱, 프로젝트 산출물이 한데 들어 있어 가장 큰 정리 대상입니다.
- `.ipynb_checkpoints/`, `__pycache__/`, `.pyc` 파일이 많이 포함되어 있습니다. 이번에는 삭제하지 않았고, 추후 정리 후보로만 둡니다.
- `ECONOMY_PROJECT/`는 이미 `data/`, `notebooks/`, `reports/`, `src/` 폴더가 있어 프로젝트형 구조로 발전시키기 쉽습니다.

## 이동 제안 목록

실제 이동은 하지 않았습니다. 다음 단계에서 필요할 때만 별도 커밋으로 진행하는 것을 권장합니다.

| 현재 위치 | 제안 위치 | 이유 |
| --- | --- | --- |
| `Country_Economy_Project/` | `projects/country-economy-analysis/` | 독립 분석 프로젝트로 관리하기 좋음 |
| `ECONOMY_PROJECT/` | `projects/exchange-rate-analysis/` | 경제 지표/환율 예측 프로젝트로 명확화 |
| `World_Happiness/` | `projects/world-happiness-analysis/` | README와 requirements가 있어 프로젝트 단위로 적합 |
| `AI_PROJECTS/linear_regression_project.ipynb` | `experiments/linear-regression-practice/` | 단일 노트북 실험으로 보임 |
| `short_code/` | `experiments/short-code/` | 짧은 실습 코드 묶음 |
| `vibe_coding/` | `experiments/vibe-coding/` | 단일 노트북 실험 |
| `Vibe_project/` | `experiments/vibe-project/` | Jupyter 기반 소형 제작물 묶음 |
| `AI_Study/프로젝트 1/` | `projects/power-usage-forecasting/` | 전력량 예측 프로젝트로 보임 |
| `AI_Study/프로젝트 2/` | `projects/rag-module-comparison/` | RAG 모듈 비교/최적화 프로젝트로 보임 |
| `AI_Study/chatbot_app*/` | `projects/rag-chatbot-app/` 또는 `archive/` | 중복 앱 후보라 확인 후 분리 필요 |
| `AI_Study/lease_law_app-main/` | `projects/lease-law-rag-app/` | 임대차 법률 RAG 앱 후보 |

## 신규 프로젝트 등록 템플릿

새 프로젝트를 추가할 때 아래 항목을 이 문서에 추가합니다.

| 항목 | 내용 |
| --- | --- |
| 이름 |  |
| 위치 | `projects/project-name/` |
| 목적 |  |
| 주요 기술 |  |
| 데이터 |  |
| 실행 방법 |  |
| 상태 | planned / active / paused / archived |

