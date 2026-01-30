"""
Final Unified RAG Module for Real-Estate Legal Chatbot

[융합 및 개선 사항]
1. Hybrid Search Architecture:
   - Dense: Upstage Solar Embedding (via Pinecone)
   - Sparse: BM25 (with Kiwi Morphological Analysis)
   - Fusion: RRF (Reciprocal Rank Fusion) 알고리즘 적용
2. Model Specification (User Request):
   - Query Normalization: Upstage Solar-Pro2
   - Answer Generation: OpenAI GPT-4o-mini
3. Advanced Context Processing:
   - 2-Stage Case Expansion: 판례 검색 시 메타데이터 검색 후 상위 건만 전문(Full-text) 로딩
   - Hierarchical Context: 법령 > 규칙 > 판례 순으로 위계화된 프롬프트 구성
4. Reranking: Cohere Rerank v3 (Multilingual)

[필수 의존성]
pip install langchain-core langchain-community langchain-openai langchain-upstage langchain-pinecone
pip install rank_bm25 kiwipiepy pinecone-client cohere
"""

from __future__ import annotations

import logging
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

# LangChain & Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Models (Upstage, OpenAI)
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_openai import ChatOpenAI

# Vector Stores & Retrievers
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone
import cohere

# Morphological Analyzer (for Korean BM25)
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False

# Logging Setup
logger = logging.getLogger("RAG_Pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==========================================
# 0. Constants & Prompts
# ==========================================

# 법률 용어 사전 (질문 표준화 보조용)
LEGAL_KEYWORD_MAP = {
    "집주인": "임대인", "건물주": "임대인", "세입자": "임차인", "월세입자": "임차인",
    "부동산": "공인중개사", "복비": "중개보수", "계약서": "임대차계약증서",
    "전세금": "임차보증금", "보증금": "임차보증금", "월세": "차임", "방세": "차임",
    "월세올리기": "차임증액", "인상": "증액", "월세깎기": "차임감액", "할인": "감액",
    "돈먼저받기": "우선변제권", "순위": "우선변제권", "안전장치": "대항력",
    "연장하기": "계약갱신요구권", "재계약": "계약갱신", "자동연장": "묵시적갱신",
    "방빼": "계약해지", "나가라고": "계약갱신거절", "비워달라": "명도", "이사": "주택의인도",
    "전입신고": "주민등록", "집고치기": "수선의무", "물샘": "누수", "청소비": "원상회복비용",
    "깡통전세": "전세피해", "사기": "전세사기", "조정위": "주택임대차분쟁조정위원회"
}

# 답변 생성 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 대한민국 '주택 전월세 사기 예방 및 임대차 법률 전문가 AI'입니다.
사용자의 질문에 대해 제공된 [법적 위계가 정리된 참고 문서]를 바탕으로 답변하세요.

[답변 생성 원칙]
1. **법적 위계 준수**: 
   - 반드시 [SECTION 1: 핵심 법령]의 내용을 최우선 판단 기준으로 삼으세요.
   - [SECTION 1]의 내용이 모호할 때만 [SECTION 2]와 [SECTION 3]를 보충 근거로 활용하세요.
   - 만약 [SECTION 3: 판례]가 [SECTION 1: 법령]과 다르게 해석되는 특수한 경우라면, "원칙은 법령에 따르나, 판례는 예외적으로..."라고 설명하세요.

2. **답변 구조**:
   - **핵심 결론**: 질문에 대한 결론(가능/불가능/유효/무효)을 두괄식으로 명확히 요약하세요.
   - **법적 근거**: "주택임대차보호법 제O조에 따르면..." (SECTION 1 인용)
   - **실무 절차**: 필요시 신고 방법, 서류, 수수료 등 안내 (SECTION 2 인용)
   - **참고 사례**: 유사한 상황에서의 판결이나 해석 (SECTION 3 인용)
   - **주의사항**: 강행규정 위반 시 "효력이 없다"고 경고하고, 법적 분쟁 시 전문가(변호사 등)의 확인이 필요함을 고지하세요.

[법적 위계가 정리된 참고 문서]
{context}
"""


# ==========================================
# 1. Configuration Class
# ==========================================

@dataclass
class RAGConfig:
    """RAG 파이프라인 설정 관리"""
    # API Keys (Environment Variables preferred)
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    upstage_api_key: str = field(default_factory=lambda: os.getenv("UPSTAGE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    
    # Models
    embedding_model: str = "solar-embedding-1-large-passage"
    normalization_model: str = "solar-pro2"     # User Request: Upstage Solar-Pro2
    generation_model: str = "gpt-4o-mini"      # User Request: GPT-4o-mini
    generation_temperature: float = 0.1
    
    # Retrieval Settings
    top_k_dense: int = 5    # Dense 검색 개수 (인덱스 당)
    top_k_sparse: int = 10  # Sparse 검색 개수 (전체)
    rrf_k: int = 60         # RRF 상수
    
    # Index Names
    index_names: Dict[str, str] = field(default_factory=lambda: {
        "law": "law-index-final",
        "rule": "rule-index-final",
        "case": "case-index-final"
    })

    def validate(self):
        if not self.pinecone_api_key: raise ValueError("PINECONE_API_KEY is missing.")
        if not self.upstage_api_key: raise ValueError("UPSTAGE_API_KEY is missing.")
        if not self.openai_api_key: raise ValueError("OPENAI_API_KEY is missing.")


# ==========================================
# 2. RAG Pipeline Class
# ==========================================

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.config.validate()
        
        self.bm25_retriever = None
        self.kiwi = Kiwi() if KIWI_AVAILABLE else None
        
        self._init_models()
        self._init_vector_stores()
        self._init_cohere()
        
    def _init_models(self):
        """LLM 및 Embedding 모델 초기화"""
        # Embedding (Upstage Solar)
        self.embedding = UpstageEmbeddings(
            model=self.config.embedding_model,
            upstage_api_key=self.config.upstage_api_key
        )
        
        # Normalization LLM (Upstage Solar-Pro2)
        self.normalization_llm = ChatUpstage(
            model=self.config.normalization_model,
            upstage_api_key=self.config.upstage_api_key,
            temperature=0
        )
        
        # Generation LLM (OpenAI GPT-4o-mini)
        self.generation_llm = ChatOpenAI(
            model=self.config.generation_model,
            openai_api_key=self.config.openai_api_key,
            temperature=self.config.generation_temperature
        )
        
    def _init_vector_stores(self):
        """Pinecone Vector Stores 연결"""
        logger.info("🔗 Pinecone 인덱스 연결 시도...")
        self.pc = Pinecone(api_key=self.config.pinecone_api_key)
        self.stores = {}
        
        for key, index_name in self.config.index_names.items():
            try:
                self.stores[key] = PineconeVectorStore(
                    index_name=index_name,
                    embedding=self.embedding,
                    pinecone_api_key=self.config.pinecone_api_key
                )
            except Exception as e:
                logger.error(f"❌ Index '{index_name}' 연결 실패: {e}")
                
    def _init_cohere(self):
        """Cohere Rerank 클라이언트 초기화"""
        if self.config.cohere_api_key:
            self.cohere_client = cohere.Client(api_key=self.config.cohere_api_key)
            logger.info("✅ Cohere Rerank 활성화됨")
        else:
            self.cohere_client = None
            logger.warning("⚠️ Cohere API Key 없음. Rerank 기능 비활성화.")

    # ---------------------------------------------------------
    # BM25 Logic (Sparse Retrieval)
    # ---------------------------------------------------------
    def _kiwi_tokenizer(self, text: str) -> List[str]:
        """한국어 형태소 분석 토크나이저"""
        if self.kiwi:
            return [token.form for token in self.kiwi.tokenize(text)]
        return text.split()

    def build_bm25(self, documents: List[Document]):
        """
        외부 문서 리스트를 받아 로컬 BM25 인덱스 생성
        Note: 서버 시작 시 전체 문서(Law+Rule+Case)를 로드해서 호출해야 함
        """
        if not documents:
            logger.warning("⚠️ BM25 빌드 실패: 문서 리스트가 비어있음.")
            return

        logger.info(f"🏗️ BM25 인덱스 빌드 시작 (문서 {len(documents)}개)...")
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=self._kiwi_tokenizer if KIWI_AVAILABLE else None
        )
        self.bm25_retriever.k = self.config.top_k_sparse
        logger.info("✅ BM25 인덱스 빌드 완료")

    # ---------------------------------------------------------
    # Helper: RRF (Reciprocal Rank Fusion)
    # ---------------------------------------------------------
    def _apply_rrf(self, dense_results: List[Document], sparse_results: List[Document]) -> List[Document]:
        """
        Dense 결과와 Sparse 결과를 RRF 알고리즘으로 통합
        Score = 1 / (k + rank)
        """
        rrf_score_map = {}

        # 1. Dense Score 계산
        for rank, doc in enumerate(dense_results):
            # chunk_id를 고유 키로 사용 (없으면 content 일부)
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            score = 1 / (self.config.rrf_k + rank + 1)
            
            if doc_id not in rrf_score_map:
                rrf_score_map[doc_id] = {"doc": doc, "score": 0.0}
            rrf_score_map[doc_id]["score"] += score

        # 2. Sparse Score 계산 (가산)
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            score = 1 / (self.config.rrf_k + rank + 1)
            
            if doc_id not in rrf_score_map:
                rrf_score_map[doc_id] = {"doc": doc, "score": 0.0}
            rrf_score_map[doc_id]["score"] += score

        # 3. 정렬 및 리스트 변환
        sorted_items = sorted(rrf_score_map.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items]

    # ---------------------------------------------------------
    # Retrieval Logic
    # ---------------------------------------------------------
    def _get_full_case_context(self, case_no: str) -> str:
        """판례 사건번호로 전문(Full Text) 조회"""
        try:
            # Upstage Embedding requires non-empty query
            results = self.stores['case'].similarity_search(
                query="판례 전문 검색", 
                k=50, 
                filter={"case_no": {"$eq": case_no}}
            )
            # chunk_id 순 정렬
            sorted_docs = sorted(results, key=lambda x: x.metadata.get('chunk_id', ''))
            
            # 중복 제거 및 병합
            seen = set()
            unique_contents = []
            for doc in sorted_docs:
                cid = doc.metadata.get('chunk_id')
                if cid and cid not in seen:
                    unique_contents.append(doc.page_content)
                    seen.add(cid)
            
            return "\n".join(unique_contents)
        except Exception as e:
            logger.warning(f"⚠️ 판례 확장 실패 ({case_no}): {e}")
            return ""

    def triple_hybrid_retrieval(self, query: str) -> List[Document]:
        """
        [Hybrid Retrieval Strategy]
        1. Dense Search: Law, Rule, Case 인덱스 병렬 검색
        2. Sparse Search: BM25 검색 (전체 대상)
        3. Fusion: RRF 알고리즘으로 통합
        4. Expansion: 판례(Case)인 경우 전문 확장
        5. Rerank: Cohere로 최종 순위 재조정
        """
        logger.info(f"🔍 [Hybrid 검색] Query: {query}")

        # 1. Dense Search (Pinecone)
        docs_law = self.stores['law'].similarity_search(query, k=self.config.top_k_dense)
        docs_rule = self.stores['rule'].similarity_search(query, k=self.config.top_k_dense)
        docs_case = self.stores['case'].similarity_search(query, k=self.config.top_k_dense * 2)
        dense_results = docs_law + docs_rule + docs_case
        
        # 2. Sparse Search (BM25)
        sparse_results = []
        if self.bm25_retriever:
            sparse_results = self.bm25_retriever.invoke(query)
            logger.info(f"  - Dense: {len(dense_results)}건, Sparse: {len(sparse_results)}건")
        
        # 3. Fusion (RRF)
        fused_docs = self._apply_rrf(dense_results, sparse_results)
        
        # 4. Case Expansion (Top N 후보에 대해 수행)
        # Rerank 비용 절감을 위해 상위 20개 정도만 확장 고려
        candidates = fused_docs[:20]
        final_candidates = []
        seen_cases = set()

        for doc in candidates:
            case_no = doc.metadata.get('case_no')
            
            # 판례이고 아직 확장하지 않은 경우
            if case_no:
                if case_no not in seen_cases:
                    full_text = self._get_full_case_context(case_no)
                    if full_text:
                        # 전문으로 교체 (메타데이터 유지)
                        new_doc = Document(
                            page_content=f"[판례 전문: {doc.metadata.get('title')}]\n{full_text}",
                            metadata=doc.metadata
                        )
                        final_candidates.append(new_doc)
                        seen_cases.add(case_no)
            else:
                # 법령/규칙은 그대로 사용
                final_candidates.append(doc)

        # 5. Rerank (Cohere)
        if self.cohere_client and final_candidates:
            try:
                docs_content = [d.page_content for d in final_candidates]
                rerank_results = self.cohere_client.rerank(
                    model="rerank-multilingual-v3.0",
                    query=query,
                    documents=docs_content,
                    top_n=len(final_candidates)
                )
                
                reranked_docs = []
                logger.info("📊 Rerank Scores (Top 5):")
                for i, r in enumerate(rerank_results.results):
                    if r.relevance_score > 0.10:  # Threshold
                        doc = final_candidates[r.index]
                        reranked_docs.append(doc)
                        if i < 5:
                            logger.info(f"  - [{r.relevance_score:.4f}] {doc.metadata.get('title')}")
                return reranked_docs
                
            except Exception as e:
                logger.error(f"⚠️ Rerank Failed: {e}")
                return final_candidates

        return final_candidates

    # ---------------------------------------------------------
    # Context Processing & Generation
    # ---------------------------------------------------------
    def normalize_query(self, user_query: str) -> str:
        """
        [Model: Upstage Solar-Pro2]
        사용자 질문을 법률 용어로 표준화
        """
        prompt = ChatPromptTemplate.from_template("""
        당신은 법률 AI 챗봇의 전처리 담당자입니다.
        아래 [용어 사전]을 참고하여 사용자의 질문을 '법률 표준어'로 변환해 주세요.
        
        [용어 사전]
        {dictionary}
        
        [지침]
        1. 사전의 단어가 질문에 있다면 반드시 법률 용어로 변경하세요.
        2. 조사나 서술어를 문맥에 맞게 자연스럽게 수정하세요.
        3. 오직 '변경된 질문' 텍스트만 출력하세요.
        
        사용자 질문: {question}
        변경된 질문:""")
        
        chain = prompt | self.normalization_llm | StrOutputParser()
        
        try:
            return chain.invoke({"dictionary": LEGAL_KEYWORD_MAP, "question": user_query}).strip()
        except Exception as e:
            logger.warning(f"⚠️ Normalization 실패: {e}")
            return user_query

    def format_context_with_hierarchy(self, docs: List[Document]) -> str:
        """
        검색된 문서를 법적 위계(Priority)에 따라 섹션별로 재구성
        Priority: 1(최상위) -> 11(최하위)
        """
        sorted_docs = sorted(docs, key=lambda x: int(x.metadata.get('priority', 99)))
        
        sections = {1: [], 2: [], 3: []}
        for doc in sorted_docs:
            p = int(doc.metadata.get('priority', 99))
            src = doc.metadata.get('src_title', '자료')
            title = doc.metadata.get('title', '')
            # 가독성을 위해 본문 길이 일부 제한 가능하지만, 여기선 전문 포함
            entry = f"[{src}] {title}\n{doc.page_content}"
            
            if p in [1, 2, 4, 5]:      # 법률, 시행령
                sections[1].append(entry)
            elif p in [3, 6, 7, 8, 11]: # 규칙, 조례, 소송절차
                sections[2].append(entry)
            else:                       # 판례 (9), 기타
                sections[3].append(entry)
            
        context = ""
        if sections[1]: 
            context += "## [SECTION 1: 핵심 법령 (최우선 법적 근거)]\n" + "\n\n".join(sections[1]) + "\n\n"
        if sections[2]: 
            context += "## [SECTION 2: 관련 규정 및 절차 (세부 기준)]\n" + "\n\n".join(sections[2]) + "\n\n"
        if sections[3]: 
            context += "## [SECTION 3: 판례 및 해석 사례 (적용 예시)]\n" + "\n\n".join(sections[3]) + "\n\n"
            
        return context

    def generate_answer(self, user_input: str, *, skip_normalization: bool = False) -> str:
        """
        [Model: GPT-4o-mini]
        최종 답변 생성 파이프라인
        """
        # 1. Normalize
        normalized_query = user_input if skip_normalization else self.normalize_query(user_input)
        if not skip_normalization:
            logger.info(f"🔄 표준화된 질문: {normalized_query}")

        # 2. Hybrid Retrieval (Dense + Sparse + RRF + Rerank)
        retrieved_docs = self.triple_hybrid_retrieval(normalized_query)
        
        if not retrieved_docs:
            return "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다."

        # 3. Context Formatting
        hierarchical_context = self.format_context_with_hierarchy(retrieved_docs)

        # 4. Generate Answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        
        chain = prompt | self.generation_llm | StrOutputParser()

        logger.info("🤖 답변 생성 중 (Model: GPT-4o-mini)...")
        try:
            return chain.invoke({"context": hierarchical_context, "question": normalized_query}).strip()
        except Exception as e:
            logger.error(f"⚠️ 답변 생성 에러: {e}")
            return "죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다."


# ==========================================
# Main Execution Block (Example)
# ==========================================
if __name__ == "__main__":
    # 1. 환경 변수 체크
    if not os.getenv("UPSTAGE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("❌ UPSTAGE_API_KEY 또는 OPENAI_API_KEY가 설정되지 않았습니다.")
        exit(1)

    # 2. 파이프라인 초기화
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    
    # 3. BM25 인덱스 빌드 (테스트용 더미 데이터)
    # 실제 환경에서는 DB나 CSV에서 전체 법률/판례 텍스트를 로드해서 주입해야 합니다.
    print("🏗️ BM25 테스트용 데이터 빌드...")
    dummy_docs = [
        Document(page_content="주택임대차보호법 제3조(대항력) 임차인이 주택의 인도와 주민등록을 마친 때에는...", metadata={"chunk_id": "LAW_001", "priority": 1}),
        Document(page_content="확정일자 부여 및 정보제공에 관한 규칙 제2조(수수료) 수수료는 600원이다.", metadata={"chunk_id": "RULE_001", "priority": 3}),
        Document(page_content="[판례] 보증금 반환 의무와 목적물 반환 의무는 동시이행 관계에 있다.", metadata={"chunk_id": "CASE_001", "priority": 9, "case_no": "2023다12345"}),
    ]
    pipeline.build_bm25(dummy_docs)
    
    # 4. 질문 테스트
    test_query = "확정일자 받는데 수수료 얼마야? 그리고 집주인 실거주 확인은 어떻게 해?"
    print("\n" + "="*60)
    print(f"Q: {test_query}")
    print("-" * 60)
    
    answer = pipeline.generate_answer(test_query)
    
    print(f"A:\n{answer}")
    print("="*60)