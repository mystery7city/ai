# Roadmap

`ai` 폴더를 여러 AI 프로젝트를 관리하는 작업 공간으로 정리하기 위한 단계별 계획입니다.

## 1단계: 문서 기반 정리

- [x] 현재 최상위 폴더와 파일 유형 파악
- [x] `README.md` 추가
- [x] `docs/project-index.md` 작성
- [x] `docs/roadmap.md` 작성
- [x] `docs/codex-instructions.md` 작성
- [x] 관리용 기본 폴더 생성: `projects`, `experiments`, `prompt-decks`, `scripts`, `assets`, `archive`

## 2단계: 이동 전 검토

- [ ] `AI_Study/` 내부에서 실제 프로젝트와 단순 학습 자료를 분리
- [ ] 노트북, 데이터, 모델 파일 중 Git에 남길 것과 외부 보관할 것을 구분
- [ ] `.pyc`, `__pycache__`, `.ipynb_checkpoints` 정리 여부 결정
- [ ] 프로젝트별 실행 방법과 의존성 확인
- [ ] 대용량 파일은 Git LFS 사용 여부 확인

## 3단계: 프로젝트 구조화

- [ ] `projects/` 아래로 독립 프로젝트를 하나씩 이동
- [ ] 각 프로젝트에 `README.md` 작성
- [ ] 각 프로젝트별 `requirements.txt` 또는 환경 파일 정리
- [ ] 공용 스크립트는 `scripts/`로 분리
- [ ] 공용 프롬프트는 `prompt-decks/`로 분리

## 4단계: 유지보수 규칙

- [ ] 새 프로젝트 생성 템플릿 확정
- [ ] 데이터/모델 파일 커밋 기준 정리
- [ ] 프로젝트 상태값 관리: `planned`, `active`, `paused`, `archived`
- [ ] 월 1회 `docs/project-index.md` 갱신

## 우선순위 제안

1. `World_Happiness/`를 첫 구조화 대상으로 삼습니다. 파일 수가 적고 README가 이미 있어 안전합니다.
2. `ECONOMY_PROJECT/`를 두 번째 대상으로 삼습니다. 이미 `data`, `notebooks`, `reports`, `src` 구조가 있습니다.
3. `AI_Study/프로젝트 1`, `AI_Study/프로젝트 2`는 이름을 먼저 확정한 뒤 프로젝트로 승격합니다.
4. `AI_Study/` 전체는 마지막에 다룹니다. 파일 수가 많고 앱/실습/산출물이 섞여 있어 작은 단위로 나누는 편이 안전합니다.

