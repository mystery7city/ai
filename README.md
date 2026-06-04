# AI Workspace

이 폴더는 여러 AI 학습, 실험, 프로젝트를 한 저장소 안에서 관리하기 위한 작업 공간입니다.

기존 파일과 폴더는 삭제하거나 이동하지 않았습니다. 현재 구조를 먼저 문서화하고, 앞으로 새 프로젝트를 `projects/` 아래에 추가할 수 있도록 기본 관리 구조를 잡았습니다.

## 현재 상태 요약

- 기존 루트 문서: `README.md.md`, `requirements.txt`
- 주요 기존 폴더: `AI_Study`, `AI_PROJECTS`, `Country_Economy_Project`, `ECONOMY_PROJECT`, `World_Happiness`, `Vibe_project`, `vibe_coding`, `short_code`
- 새 관리 폴더: `docs`, `projects`, `experiments`, `prompt-decks`, `scripts`, `assets`, `archive`

## 권장 구조

```text
ai/
├─ README.md
├─ docs/
│  ├─ project-index.md
│  ├─ roadmap.md
│  └─ codex-instructions.md
├─ projects/
├─ experiments/
├─ prompt-decks/
├─ scripts/
├─ assets/
└─ archive/
```

## 폴더 역할

| 폴더 | 역할 |
| --- | --- |
| `docs/` | 프로젝트 인덱스, 로드맵, 작업 규칙 문서 |
| `projects/` | 완성도 있게 관리할 개별 AI 프로젝트 |
| `experiments/` | 짧은 실험, 검증 코드, 임시 노트북 |
| `prompt-decks/` | 프롬프트 묶음, 템플릿, 비교 실험 |
| `scripts/` | 반복 작업 자동화 스크립트 |
| `assets/` | 공용 이미지, 샘플 데이터 설명, 발표 자료 등 |
| `archive/` | 더 이상 활성 관리하지 않는 자료의 보관 후보 |

## 새 프로젝트 추가 기준

새 프로젝트는 가능하면 아래 구조로 `projects/` 안에 추가합니다.

```text
projects/project-name/
├─ README.md
├─ docs/
├─ notebooks/
├─ src/
├─ data/
├─ outputs/
└─ requirements.txt
```

대용량 데이터, 모델 파일, 실행 결과는 바로 커밋하기 전에 Git LFS 또는 별도 보관 여부를 확인합니다.

## 문서

- [프로젝트 인덱스](docs/project-index.md)
- [로드맵](docs/roadmap.md)
- [Codex 작업 지침](docs/codex-instructions.md)

