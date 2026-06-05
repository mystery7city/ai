# Sample Output

아래는 가짜 예시다. 실제 파일 경로나 키 값은 포함하지 않는다.

```text
AI Study Repository Cleaner
===========================
Root: C:/example/ai-study
Generated: 2026-06-05T10:00:00
Total files: 1234
Total folders: 120
Total size: 512.30 MB
Git exclude candidates: 87
Sensitive candidates: 6 (values are never printed)
Large file candidates: 9
Markdown report: C:/example/ai-study/reports/repo_cleaner_report.md
JSON report: C:/example/ai-study/reports/repo_cleaner_report.json
```

## Markdown Report Preview

```markdown
# AI Study Repository Cleaner Report

생성일: 2026-06-05T10:00:00

## 1. 전체 요약
- 총 파일 수: 1234
- 총 폴더 수: 120
- 전체 용량: 512.30 MB
- Git 제외 후보: 87
- 민감정보 후보: 6
- 대용량 파일 후보: 9

## 2. 최상위 폴더별 요약

| 폴더 | 분류 | 파일 수 | 하위 폴더 수 | 용량 | 주요 확장자 | 위험 후보 | 대용량 |
|---|---|---:|---:|---:|---|---:|---:|
| `example_project` | AI/LLM/RAG | 120 | 14 | 80.20 MB | .py:50, .ipynb:8 | 3 | 1 |

## 3. Git 제외 후보
- `example_project/.ipynb_checkpoints` - jupyter checkpoint
- `example_project/model.h5` - model file

## 4. 민감정보 후보
주의: 실제 키/비밀번호 값은 출력하지 않음.
- `example_project/settings.py` - content keywords: SECRET_KEY; 값은 출력하지 않음

## 5. 대용량 파일 후보
| 경로 | 크기 | 확장자 | 추천 조치 |
|---|---:|---|---|
| `example_project/data/train.csv` | 42.00 MB | .csv | Git LFS / 외부 보관 / 수동 확인 |
```

