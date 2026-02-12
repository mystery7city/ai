<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,2,5,30&height=200&section=header&text=F-NAL&fontSize=80&fontAlignY=35&desc=주택임대차%20법률%20RAG%20AI%20상담%20시스템%20|%20AWS%20배포&descAlignY=55&descAlign=50)

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&pause=1000&color=4A90E2&center=true&vCenter=true&width=700&lines=Hybrid+RAG+Legal+Chatbot;AWS+EC2+Production+Deployment;Django+%2B+Streamlit+Dual+Server)](https://git.io/typing-svg)

<br>

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-52.79.175.135-FF6B6B?style=for-the-badge)](http://52.79.175.135)
[![YouTube](https://img.shields.io/badge/📺_시연_영상-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=GFpilFkehSo)
[![Original Repo](https://img.shields.io/badge/📦_원본_Repository-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thre3o2wo/2ndTeamProject)

</div>

---

## 📌 개요

이 리포지토리는 **F-NAL 법률 RAG AI 챗봇**의 **AWS EC2 배포 버전**입니다.

전체 프로젝트 문서(파이프라인 상세, 모듈 API, 설정 커스터마이징 등)는 [원본 Repository](https://github.com/thre3o2wo/2ndTeamProject)를 참고해 주세요.

> **프로젝트명**: 법률 RAG 기반 주택 임대차계약 리스크 분석 AI 챗봇  
> **팀**: TEAM 안전한家 (박상용, 김재학, 김지훈, 김효경)  
> **개발 기간**: 2026.01.12 - 2026.02.10 (4주)  
> **과정**: [KDT] 기업맞춤형 AI+X 융복합 인재 양성 과정 2차 팀프로젝트

---

## 🔧 기술 스택

<div align="center">

<table>
  <tr>
    <td width="160px" align="center"><b>구분</b></td>
    <td><b>기술</b></td>
  </tr>
  <tr>
    <td align="center"><b>AI / LLM</b></td>
    <td>
      <img src="https://img.shields.io/badge/Upstage-SOLAR-6E3FF3?style=for-the-badge&logo=lightning&logoColor=white">
      <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white">
      <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Data & Vector</b></td>
    <td>
      <img src="https://img.shields.io/badge/Pinecone-VectorDB-243A5E?style=for-the-badge&logo=pinecone&logoColor=white">
      <img src="https://img.shields.io/badge/Cohere-Rerank-3FB58E?style=for-the-badge&logo=cohere&logoColor=white">
      <img src="https://img.shields.io/badge/BM25_Sparse-02569B?style=for-the-badge">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Document & OCR</b></td>
    <td>
      <img src="https://img.shields.io/badge/EasyOCR-5DADE2?style=for-the-badge&logo=googlelens&logoColor=white">
      <img src="https://img.shields.io/badge/pytesseract-00A4EF?style=for-the-badge&logo=googlelens&logoColor=white">
      <img src="https://img.shields.io/badge/pdfplumber-FF0000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white">
      <img src="https://img.shields.io/badge/PyMuPDF-E0115F?style=for-the-badge&logo=adobeacrobatreader&logoColor=white">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Web Framework</b></td>
    <td>
      <img src="https://img.shields.io/badge/Django_5.2-092E20?style=for-the-badge&logo=django&logoColor=white">
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Deployment</b></td>
    <td>
      <img src="https://img.shields.io/badge/AWS_EC2-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white">
    </td>
  </tr>
  <tr>
    <td align="center"><b>Version Control</b></td>
    <td>
      <img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
      <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
    </td>
  </tr>
</table>

</div>

---

## 🏗️ 배포 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                     AWS EC2 Instance                     │
│                                                          │
│  ┌─────────────────┐         ┌─────────────────────────┐ │
│  │   Django App     │         │   Streamlit App         │ │
│  │   (Port 8000)    │         │   (Port 8501)           │ │
│  │                  │         │                         │ │
│  │  · 메인 페이지    │  ←───→  │  · 채팅 인터페이스      │ │
│  │  · Chat REST API │         │  · 파일 업로드 / OCR    │ │
│  │  · OCR 세션 관리  │         │  · RAG 파이프라인       │ │
│  └────────┬─────────┘         └────────────┬────────────┘ │
│           └──────────┬─────────────────────┘              │
│                      │                                    │
│             ┌────────▼─────────┐                          │
│             │    modules/      │                          │
│             │  ├─ rag_module   │  ← Hybrid RAG Pipeline  │
│             │  └─ ocr_module   │  ← OCR / PDF 처리       │
│             └────────┬─────────┘                          │
└──────────────────────│────────────────────────────────────┘
                       │
          ┌────────────┼──────────────────┐
          ▼            ▼                  ▼
   ┌───────────┐ ┌───────────┐    ┌────────────┐
   │ Pinecone  │ │  Upstage  │    │   OpenAI   │
   │ Vector DB │ │ SOLAR Pro2│    │ GPT-4o-mini│
   │           │ │ Embeddings│    │            │
   │ law-index │ └───────────┘    └────────────┘
   │ rule-index│       ┌───────────┐
   │ case-index│       │  Cohere   │
   └───────────┘       │  Reranker │
                       └───────────┘
```

---

## 📁 배포 폴더 구조

> 원본 리포에서 **배포에 필요한 파일만** 추출한 구조입니다.

```
lease_law_app/
│
├── .git/                        # Git 버전 관리
├── .venv/                       # Python 가상환경
├── .gitignore
├── .env                         # 환경 변수 (API 키)
├── requirements.txt             # Python 의존성
├── web_chatbot.py               # Streamlit 채팅 인터페이스
│
├── modules/                     # 공유 핵심 모듈
│   ├── rag_module.py            #   Hybrid RAG 파이프라인
│   └── ocr_module.py            #   OCR 유틸리티
│
└── chatbot_app/                 # Django 웹앱
    ├── manage.py
    ├── config/                  #   Django 설정
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    └── chatbot/                 #   챗봇 앱
        ├── views.py             #     Chat API, OCR 엔드포인트
        ├── urls.py
        └── templates/chatbot/
            └── index.html       #     메인 페이지
```

---

## 📊 성능 평가 (RAGAS)

<div align="center">

<table>
  <tr>
    <th align="center">지표</th>
    <th align="center">점수</th>
    <th align="center">의미</th>
  </tr>
  <tr>
    <td align="center"><b>✅ Context Precision</b></td>
    <td align="center"><b>1.00</b></td>
    <td align="center">검색 정밀도 — Cohere Reranker 적용</td>
  </tr>
  <tr>
    <td align="center"><b>✅ Context Recall</b></td>
    <td align="center"><b>0.75</b></td>
    <td align="center">검색 재현율 — Hybrid Retrieval</td>
  </tr>
  <tr>
    <td align="center"><b>✅ Ri (Custom)</b></td>
    <td align="center"><b>0.85</b></td>
    <td align="center">종합 검색 성능 지표</td>
  </tr>
  <tr>
    <td align="center">Faithfulness</td>
    <td align="center">0.34</td>
    <td align="center">답변 충실성</td>
  </tr>
  <tr>
    <td align="center">Answer Relevancy</td>
    <td align="center">0.41</td>
    <td align="center">답변 관련성</td>
  </tr>
</table>

</div>

### 주요 성과

| 성과 | 설명 |
|:---:|:---|
| 🎯 **Precision 100%** | Cohere Reranker 적용으로 완벽한 검색 정밀도 달성 |
| 🔀 **Hybrid 검색** | Dense + Sparse 통합 파이프라인으로 안정적 Recall 확보 |
| 🤖 **LLM 최적화** | 6개 모델 비교 평가 → GPT-4o-mini 선정 (속도/비용/품질 최적) |
| 📝 **Prompt 분리** | 일반 상담 vs 계약서 분석 모드 이원화로 실용성 향상 |
| 🌐 **AWS 배포** | EC2 인스턴스에 Django + Streamlit 듀얼 서버 운영 |

---

## ⚙️ 환경 변수

`.env` 파일을 프로젝트 루트(`lease_law_app/`)에 생성합니다.

```env
# LLM & Embedding
OPENAI_API_KEY=sk-proj-xxxxx
UPSTAGE_API_KEY=up_xxxxx

# Vector DB
PINECONE_API_KEY=pcsk_xxxxx

# Reranking
COHERE_API_KEY=xxxxx

# Django
DJANGO_SECRET_KEY=your-secret-key-here

# HuggingFace (선택)
HF_TOKEN=hf_xxxxx
```

**API 키 발급:**
| 서비스 | 용도 | 발급 링크 |
|--------|------|-----------|
| **Upstage** | 임베딩, 질문 표준화 | https://console.upstage.ai |
| **Pinecone** | 벡터 스토어 | https://app.pinecone.io |
| **Cohere** | 리랭킹 | https://dashboard.cohere.com |
| **OpenAI** | 답변 생성 | https://platform.openai.com |

---

## 🚀 AWS EC2 배포 가이드

### 1️⃣ EC2 인스턴스 준비

```bash
# EC2 접속
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 및 필수 패키지 설치
sudo apt install -y python3-pip python3-venv git
```

### 2️⃣ Tesseract OCR 설치

```bash
# OCR 기능에 필요 (한국어 언어팩 포함)
sudo apt install -y tesseract-ocr tesseract-ocr-kor
```

### 3️⃣ 프로젝트 클론 & 환경 설정

```bash
# 리포지토리 클론
git clone <REPOSITORY_URL>
cd lease_law_app

# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
nano .env    # API 키 입력
```

### 4️⃣ Django 초기 설정

```bash
cd chatbot_app

# DB 마이그레이션
python manage.py migrate

# 정적 파일 수집
python manage.py collectstatic --noinput
```

### 5️⃣ 서버 실행

```bash
# Django 서버 (백그라운드)
nohup python manage.py runserver 0.0.0.0:8000 > django.log 2>&1 &

# 프로젝트 루트로 이동 후 Streamlit 실행
cd ..
nohup streamlit run web_chatbot.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    > streamlit.log 2>&1 &
```

### 6️⃣ EC2 보안 그룹 (인바운드 규칙)

<div align="center">

| 포트 | 프로토콜 | 소스 | 용도 |
|:---:|:---:|:---:|:---|
| **22** | TCP | My IP | SSH 접속 |
| **8000** | TCP | 0.0.0.0/0 | Django 메인 서버 |
| **8501** | TCP | 0.0.0.0/0 | Streamlit 챗봇 |

</div>

### 7️⃣ 접속 확인

```
🌐 Django 메인 :  http://<EC2_PUBLIC_IP>:8000/
💬 Streamlit   :  http://<EC2_PUBLIC_IP>:8501/
```

---

## 📡 API 엔드포인트

### `POST /api/chat/`

**텍스트 질문 (JSON):**
```bash
curl -X POST http://<EC2_PUBLIC_IP>:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "전세 보증금 반환 절차는 어떻게 되나요?"}'
```

**파일 첨부 질문 (Multipart):**
```bash
curl -X POST http://<EC2_PUBLIC_IP>:8000/api/chat/ \
  -F "message=이 계약서에서 문제점이 있나요?" \
  -F "files=@contract.pdf"
```

**응답:**
```json
{
  "normalized_query": "임대차보증금 반환 청구 절차",
  "references": ["주택임대차보호법 제3조(대항력 등)", ...],
  "response": "주택임대차보호법 제3조에 따르면...",
  "has_ocr_context": false
}
```

### `POST /api/clear-ocr/`

세션에 저장된 OCR 컨텍스트를 삭제합니다.

---

## 🛠️ 운영 관리

### 로그 확인
```bash
# Django 로그
tail -f django.log

# Streamlit 로그
tail -f streamlit.log
```

### 서버 재시작
```bash
# 기존 프로세스 종료
pkill -f "manage.py runserver"
pkill -f "streamlit run"

# 재시작
cd ~/lease_law_app/chatbot_app
nohup python manage.py runserver 0.0.0.0:8000 > django.log 2>&1 &

cd ~/lease_law_app
nohup streamlit run web_chatbot.py \
    --server.port 8501 --server.address 0.0.0.0 \
    > streamlit.log 2>&1 &
```

### 프로세스 상태 확인
```bash
ps aux | grep -E "(manage.py|streamlit)"
```

---

## 🔗 관련 링크

<div align="center">

| 링크 | 설명 |
|:---:|:---|
| [📦 원본 Repository](https://github.com/thre3o2wo/2ndTeamProject) | 전체 프로젝트 문서, 파이프라인 상세, 설정 가이드 |
| [📺 시연 영상](https://www.youtube.com/watch?v=GFpilFkehSo) | 프로젝트 데모 영상 |
| [🌐 Live Demo](http://52.79.175.135) | AWS 배포 서비스 |

</div>

---

<div align="center">

![footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,2,5,30&height=100&section=footer)

</div>
