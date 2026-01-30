"""
Unified RAG module with Hybrid Search (Dense + BM25)

주택임대차 RAG 시스템 - 하이브리드 검색 및 답변 생성 모듈

[주요 기능]
1. 사용자 질문 표준화 (normalize_query)
2. Hybrid Retrieval: Dense(Solar) + Sparse(BM25)
3. 2-Stage Case Expansion: 판례 검색 효율화 및 전문 확장
4. Rerank: Cohere 기반 정밀 재정렬
5. 최종 답변 생성: 법적 위계(Priority) 반영

[필수 의존성]
pip install rank_bm25 kiwipiepy langchain-upstage langchain-pinecone cohere
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# LangChain Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Models & Vector Stores
from langchain_ollama import ChatOllama
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone
import cohere

# 형태소 분석기 (BM25용)
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("⚠️ Warning: 'kiwipiepy' not installed. BM25 will use simple whitespace tokenizer.")

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ==========================================
# 0. Constants & Config
# ==========================================

# 법률 용어 사전 (질문 표준화용)
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

# LLM 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 대한민국 '주택 전월세 사기 예방 및 임대차 법률 전문가 AI'입니다.
사용자의 질문에 대해 제공된 [법적 위계가 정리된 참고 문서]를 바탕으로 답변하세요.

[답변 생성 원칙]
1. **법적 위계 준수**: 
   - 반드시 [SECTION 1: 핵심 법령]의 내용을 최우선 판단 기준으로 삼으세요.
   - [SECTION 1]의 내용이 모호할 때만 [SECTION 2]와 [SECTION 3]를 보충 근거로 활용하세요.
   - 만약 [SECTION 3: 판례]가 [SECTION 1: 법령]과 다르게 해석되는 특수한 경우라면, "원칙은 법령에 따르나, 판례는 예외적으로..."라고 설명하세요.

2. **답변 구조**:
   - **핵심 결론**: 질문에 대한 결론(가능/불가능/유효/무효)을 두괄식으로 요약.
   - **법적 근거**: "주택임대차보호법 제O조에 따르면..." (SECTION 1 인용)
   - **실무 절차**: 필요시 신고 방법, 서류 등 안내 (SECTION 2 인용)
   - **참고 사례**: 유사한 상황에서의 판결이나 해석 (SECTION 3 인용)
   - **주의사항**: 강행규정 위반 시 "효력이 없다"고 경고하고, 최종적으로 전문가 확인이 필요함을 고지하세요.

[법적 위계가 정리된 참고 문서]
{context}
"""

@dataclass
class RAGConfig:
    # API Keys (환경 변수 또는 직접 입력)
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    upstage_api_key: str = os.getenv("UPSTAGE_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    
    # Models
    embedding_model: str = "solar-embedding-1-large-passage"
    llm_model: str = "exaone3.5:2.4b"
    llm_temperature: float = 0.1
    
    # Index Names
    index_names: Dict[str, str] = None

    def __post_init__(self):
        if self.index_names is None:
            self.index_names = {
                "law": "law-index-final",
                "rule": "rule-index-final",
                "case": "case-index-final"
            }


# ==========================================
# 1. RAG Pipeline Class
# ==========================================
class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self._init_components()
        self.bm25_retriever = None  # 추후 build_bm25() 호출 시 초기화

    def _init_components(self):
        """기본 컴포넌트 초기화 (Pinecone, LLM, Cohere, Kiwi)"""
        # 1. Embedding
        if not self.config.upstage_api_key:
             logger.warning("⚠️ UPSTAGE_API_KEY가 설정되지 않았습니다.")
        self.embedding = UpstageEmbeddings(model=self.config.embedding_model)
        
        # 2. Pinecone Stores (Dense)
        if not self.config.pinecone_api_key:
             raise ValueError("❌ PINECONE_API_KEY가 필수입니다.")
             
        pc = Pinecone(api_key=self.config.pinecone_api_key)
        self.stores = {}
        logger.info("🔗 Pinecone 인덱스 연결 중...")
        for key, name in self.config.index_names.items():
            try:
                self.stores[key] = PineconeVectorStore(
                    index_name=name,
                    embedding=self.embedding,
                    pinecone_api_key=self.config.pinecone_api_key
                )
            except Exception as e:
                logger.error(f"❌ Pinecone 인덱스 '{name}' 연결 실패: {e}")
        
        # 3. LLM (Ollama)
        self._generation_llm = ChatOllama(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature
        )
        
        # 4. Cohere Client (Rerank)
        if self.config.cohere_api_key:
            self.cohere_client = cohere.Client(api_key=self.config.cohere_api_key)
            logger.info("✅ Cohere Rerank 활성화됨")
        else:
            self.cohere_client = None
            logger.warning("⚠️ Cohere API Key 없음. Rerank 기능이 비활성화됩니다.")

        # 5. Kiwi Tokenizer (for BM25)
        if KIWI_AVAILABLE:
            self.kiwi = Kiwi()
            logger.info("✅ Kiwi 형태소 분석기 로드 완료")
        else:
            self.kiwi = None

    # ---------------------------------------------------------
    # BM25 Management
    # ---------------------------------------------------------
    def kiwipiepy_tokenizer(self, text: str) -> List[str]:
        """BM25용 한국어 형태소 분석 토크나이저"""
        if self.kiwi:
            return [token.form for token in self.kiwi.tokenize(text)]
        return text.split()  # Fallback: 띄어쓰기 기준

    def build_bm25(self, documents: List[Document]):
        """
        외부에서 로드한 문서 리스트로 로컬 BM25 인덱스를 생성합니다.
        (서버 시작 시점에 전체 문서를 불러와 주입해야 함)
        """
        if not documents:
            logger.warning("⚠️ BM25 빌드를 위한 문서 리스트가 비어있습니다.")
            return

        logger.info(f"🏗️ BM25 인덱스 생성 시작 (문서 수: {len(documents)}개)...")
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=self.kiwipiepy_tokenizer if KIWI_AVAILABLE else None
        )
        # BM25 검색 개수 설정 (Dense보다 조금 더 많이 가져와서 Reranker에 넘김)
        self.bm25_retriever.k = 10 
        logger.info("✅ BM25 인덱스 생성 완료!")

    # ---------------------------------------------------------
    # Retrieval Logic (Dense + Sparse)
    # ---------------------------------------------------------
    def get_full_case_context(self, case_no: str) -> str:
        """판례 전문 확장 (기존 로직 유지)"""
        try:
            # Query must not be empty for Upstage embedding
            results = self.stores['case'].similarity_search(
                query="판례 전문 검색", 
                k=50, 
                filter={"case_no": {"$eq": case_no}}
            )
            # chunk_id 순 정렬
            sorted_docs = sorted(results, key=lambda x: x.metadata.get('chunk_id', ''))
            
            seen = set()
            unique_docs = []
            for doc in sorted_docs:
                cid = doc.metadata.get('chunk_id')
                if cid and cid not in seen:
                    unique_docs.append(doc)
                    seen.add(cid)
            return "\n".join([doc.page_content for doc in unique_docs])
        except Exception as e:
            logger.warning(f"⚠️ 판례 확장 실패 ({case_no}): {e}")
            return ""

    def triple_hybrid_retrieval(self, query: str, k_dense_law=3, k_dense_case=3) -> List[Document]:
        """
        [Hybrid Search Workflow]
        1. Dense Search: Law/Rule/Case 인덱스에서 의미 검색
        2. Sparse Search (BM25): 전체 문서에서 키워드 검색
        3. Ensemble: 결과 통합 및 중복 제거
        4. Case Expansion: 판례 전문 확장
        5. Rerank: Cohere로 최종 정렬
        """
        logger.info(f"🔍 [통합 검색] 쿼리: '{query}'")

        # 1. Dense Search (Pinecone)
        docs_law = self.stores['law'].similarity_search(query, k=k_dense_law)
        docs_rule = self.stores['rule'].similarity_search(query, k=k_dense_law)
        docs_case = self.stores['case'].similarity_search(query, k=k_dense_case * 2)
        
        dense_results = docs_law + docs_rule + docs_case
        logger.info(f"  - Dense 결과: {len(dense_results)}건")
        
        # 2. Sparse Search (BM25) - 로컬 인덱스가 있는 경우만
        sparse_results = []
        if self.bm25_retriever:
            # BM25는 전체 문서에서 검색
            sparse_results = self.bm25_retriever.invoke(query)
            logger.info(f"  - BM25 결과: {len(sparse_results)}건")

        # 3. Ensemble (Union & Deduplication)
        combined_docs_map = {}
        
        # Dense 결과 우선 추가
        for doc in dense_results:
            # chunk_id가 없으면 content 앞부분을 키로 사용
            cid = doc.metadata.get('chunk_id', doc.page_content[:30])
            combined_docs_map[cid] = doc
            
        # BM25 결과 추가 (이미 있는 문서는 스킵 -> 사실상 Dense가 우선순위이나, Reranker가 판단함)
        for doc in sparse_results:
            cid = doc.metadata.get('chunk_id', doc.page_content[:30])
            if cid not in combined_docs_map:
                combined_docs_map[cid] = doc 
        
        combined_docs = list(combined_docs_map.values())
        logger.info(f"  - 통합 후보군: {len(combined_docs)}건")
        
        # 4. Case Expansion (판례 전문 확장)
        final_candidates = []
        seen_cases = set()
        
        for doc in combined_docs:
            case_no = doc.metadata.get('case_no')
            # 판례이면서 아직 확장 안 된 경우
            if case_no:
                if case_no not in seen_cases:
                    full_text = self.get_full_case_context(case_no)
                    if full_text:
                        # 원본 메타데이터 유지, 내용은 전문으로 교체
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
        if self.cohere_client:
            try:
                # 문서 내용 리스트 추출
                docs_content = [d.page_content for d in final_candidates]
                
                rerank_results = self.cohere_client.rerank(
                    model="rerank-multilingual-v3.0",
                    query=query,
                    documents=docs_content,
                    top_n=len(final_candidates)
                )
                
                reranked_docs = []
                logger.info("📊 Rerank 점수 (Top 5):")
                for i, r in enumerate(rerank_results.results):
                    if r.relevance_score > 0.10: # Threshold
                        doc = final_candidates[r.index]
                        reranked_docs.append(doc)
                        if i < 5:
                            logger.info(f"  - [{r.relevance_score:.4f}] {doc.metadata.get('title')}")
                            
                return reranked_docs
                
            except Exception as e:
                logger.error(f"⚠️ Rerank Failed: {e}")
                return final_candidates # Fallback
        
        return final_candidates

    # ---------------------------------------------------------
    # Context & Generation
    # ---------------------------------------------------------
    def normalize_query(self, user_query: str) -> str:
        """LLM을 사용하여 사용자 질문을 법률 용어로 표준화"""
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
        
        chain = prompt | self._generation_llm | StrOutputParser()
        
        try:
            return chain.invoke({"dictionary": LEGAL_KEYWORD_MAP, "question": user_query}).strip()
        except Exception as e:
            logger.warning(f"⚠️ 전처리 실패: {e}")
            return user_query

    def format_context_with_hierarchy(self, docs: List[Document]) -> str:
        """검색된 문서를 법적 위계(Priority)에 따라 섹션별로 재구성"""
        # Priority 오름차순 정렬 (1이 가장 높음)
        sorted_docs = sorted(docs, key=lambda x: int(x.metadata.get('priority', 99)))
        
        sections = {1: [], 2: [], 3: []}
        for doc in sorted_docs:
            p = int(doc.metadata.get('priority', 99))
            src = doc.metadata.get('src_title', '자료')
            title = doc.metadata.get('title', '')
            entry = f"[{src}] {title}\n{doc.page_content}"
            
            if p in [1, 2, 4, 5]: 
                sections[1].append(entry)
            elif p in [3, 6, 7, 8, 11]: 
                sections[2].append(entry)
            else: 
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
        최종 답변 생성 파이프라인 실행
        """
        # 1) Normalize
        if not skip_normalization:
            normalized_query = self.normalize_query(user_input)
            logger.info(f"🔄 표준화된 질문: {normalized_query}")
        else:
            normalized_query = user_input

        # 2) Retrieve (Hybrid)
        retrieved_docs = self.triple_hybrid_retrieval(normalized_query)
        
        if not retrieved_docs:
            return "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다."

        # 3) Context Formatting
        hierarchical_context = self.format_context_with_hierarchy(retrieved_docs)

        # 4) Generate Answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | self._generation_llm | StrOutputParser()

        logger.info("🤖 답변 생성 중...")
        try:
            return chain.invoke({"context": hierarchical_context, "question": normalized_query})
        except Exception as e:
            logger.error(f"⚠️ 답변 생성 중 에러 발생: {e}")
            return "죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다."


# ==========================================
# 실행 예시 (Testing Block)
# ==========================================
if __name__ == "__main__":
    # 1. 설정 및 초기화
    config = RAGConfig()
    try:
        pipeline = RAGPipeline(config)
        
        # 2. [중요] BM25 빌드 테스트 (실제로는 전체 문서 로드 필요)
        # 테스트용 더미 데이터 생성 (원래는 CSV 등에서 로드해야 함)
        dummy_docs = [
            Document(page_content="주택임대차보호법 제3조(대항력) 임차인이 주택의 인도와 주민등록을 마친 때에는...", metadata={"chunk_id": "LAW_001", "priority": 1}),
            Document(page_content="확정일자 부여 및 정보제공에 관한 규칙 제2조(수수료) 확정일자 부여 수수료는 건당 600원으로 한다.", metadata={"chunk_id": "RULE_001", "priority": 3}),
        ]
        pipeline.build_bm25(dummy_docs)
        
        # 3. 질문 테스트
        test_query = "확정일자 받으려면 돈 얼마나 들어?"
        print("\n" + "="*60)
        print(pipeline.generate_answer(test_query))
        print("="*60)
        
    except Exception as e:
        print(f"🔥 초기화 실패: {e}")