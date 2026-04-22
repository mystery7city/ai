# 법률 AI 상담 챗봇

주택 임대차(전월세) 법률 상담 AI 챗봇입니다. RAG(Retrieval-Augmented Generation) 기반으로 법령, 규정, 판례를 검색하고 GPT를 활용하여 답변을 생성합니다.

## 주요 기능

- **하이브리드 검색**: Dense(Pinecone) + Sparse(BM25) 검색 결합(RRF)
- **Rerank**: Cohere reranker로 검색 결과 정렬
- **OCR 지원**: PDF/이미지 파일 업로드 시 OCR로 텍스트 추출
- **법적 위계 구분**: SECTION 0(계약서) → SECTION 1(법령) → SECTION 2(규정) → SECTION 3(판례)

## 설치

### 1. 가상환경 생성 및 활성화
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. Tesseract OCR 설치 (이미지 OCR용)
- Windows: `winget install UB-Mannheim.TesseractOCR`
- 한글 데이터: `tessdata/kor.traineddata` 필요 (최상위 경로에 있음)
- "C:\Program Files\Tesseract-OCR\tessdata\" 등 경로 확인해 복사

### 4. 환경변수 설정
`.env.example`을 `.env`로 복사하고 API 키 입력:
```bash
copy .env.example chatbot_app\.env
```

## 실행

```bash
cd chatbot_app
python manage.py runserver 8000
```

브라우저에서 http://localhost:8000/ 접속

## 프로젝트 구조

```
final/
├── chatbot_app/           # Django 웹앱
│   ├── chatbot/          # 챗봇 앱
│   │   ├── views.py      # API 엔드포인트
│   │   └── templates/    # HTML 템플릿
│   ├── config/           # Django 설정
│   └── manage.py
├── modules/               # 핵심 모듈 (Django에서 참조)
│   ├── rag_module.py     # RAG 파이프라인
│   └── ocr_module.py     # OCR 유틸
├── requirements.txt
├── .env.example
└── README.md
```

## API 사용법

### 텍스트 질문
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "전세 보증금 반환 절차는?"}'
```

### 파일 첨부 질문
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -F "message=이 계약서에서 문제점이 있나요?" \
  -F "files=@contract.pdf"
```

## 팀

법률 AI 챗봇 프로젝트팀
