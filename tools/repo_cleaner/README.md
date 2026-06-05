# AI Study Repository Cleaner

이 도구는 AI / 프로그래밍 공부 저장소를 안전하게 분석하기 위한 읽기 전용 스캐너다.

## 기능

- 폴더별 파일 수/용량 분석
- Git 제외 후보 탐지
- 민감정보 후보 탐지
- 대용량 파일 탐지
- 프로젝트 분류
- Markdown/JSON 보고서 생성

## 주의

이 도구는 파일을 삭제하거나 이동하지 않는다.
실제 API Key, 비밀번호, 토큰 값은 출력하지 않는다.

민감정보 후보는 파일 경로와 위험 유형만 보고한다. `.env`, 노트북, 설정 파일을 읽더라도 실제 값은 보고서에 쓰지 않는다.

## 사용법

저장소 루트에서 실행한다.

```bash
python tools/repo_cleaner/repo_cleaner.py
python tools/repo_cleaner/repo_cleaner.py --root .
python tools/repo_cleaner/repo_cleaner.py --large-threshold-mb 10
python tools/repo_cleaner/repo_cleaner.py --json
```

## 생성 파일

실행하면 아래 보고서를 생성한다.

```text
reports/repo_cleaner_report.md
reports/repo_cleaner_report.json
```

## 기본 제외 폴더

아래 폴더는 깊은 스캔에서 제외하고, Git 제외 후보로 표시한다.

```text
.git/
venv/
.venv/
__pycache__/
.ipynb_checkpoints/
node_modules/
staticfiles/
_staticfiles/
chroma/
chroma_upstage/
downloads/
```

## 한계

- 민감정보 탐지는 키워드 기반이므로 오탐과 누락이 있을 수 있다.
- 노트북 출력 셀에 숨은 값은 키워드가 없으면 놓칠 수 있다.
- Git에서 이미 추적 중인 제외 후보는 `.gitignore`만으로 자동 해제되지 않는다.

