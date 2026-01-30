"""
RAG LLM Pipeline for Django Web Application (Unified Version)
주택임대차 RAG 시스템 - 통합 검색 및 답변 생성 모듈

[융합 출처]
- rag_module_cl.py: 타입 힌트, 파라미터 유연성, Django 통합 가이드
- rag_module_ge.py: 풍부한 키워드 사전, 검색 배수 전략
- rag_module_ge2.py: INDEX_NAMES 중앙 관리, 깔끔한 구조

[주요 기능]
1. RAGConfig: 모든 설정을 중앙에서 관리하는 설정 클래스
2. normalize_query: 사용자 질문을 법률 용어로 표준화
3. triple_hybrid_retrieval: 3중 인덱스 통합 검색 + Reranking
4. format_context_with_hierarchy: 법적 위계에 따른 컨텍스트 재정렬
5. generate_final_answer: 최종 답변 생성

[사용 예시]
    from rag_module_unified import RAGPipeline, RAGConfig
    
    # 기본 설정으로 파이프라인 생성
    pipeline = RAGPipeline()
    
    # 또는 커스텀 설정
    config = RAGConfig(llm_model="exaone3.5:7.8b", temperature=0.2)
    pipeline = RAGPipeline(config)
    
    # 답변 생성
    answer = pipeline.generate_answer("집주인이 월세를 올려달라고 해요")
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore

# Vector DB imports
from pinecone import Pinecone

# Reranking import (Optional)
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# ==========================================
# 로깅 설정
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# 0. 설정 및 상수 정의
# ==========================================

# 인덱스 이름 중앙 관리 (from ge2.py)
INDEX_NAMES: Dict[str, str] = {
    "law": "law-index-final",    # Priority 1,2,4,5: 주임법, 민법 등 핵심 법률
    "rule": "rule-index-final",  # Priority 3,6,7,8,11: 시행규칙, 조례, 절차
    "case": "case-index-final"   # Priority 9: 판례, 상담사례
}

# 주택임대차 챗봇 질문 표준화 사전 (from ge.py/cl.py - 풍부한 버전)
KEYWORD_DICT: Dict[str, str] = {
    # 1. 계약 주체 및 대상
    "집주인": "임대인", "건물주": "임대인", "주인집": "임대인", 
    "임대업자": "임대인", "새주인": "임대인",
    "세입자": "임차인", "월세입자": "임차인", "세들어사는사람": "임차인", 
    "임차자": "임차인", "입주자": "임차인",
    "부동산": "공인중개사", "중개인": "공인중개사", "중개소": "공인중개사",
    "빌라": "임차주택", "아파트": "임차주택", "오피스텔": "임차주택", 
    "우리집": "임차주택", "거주지": "임차주택",
    "계약서": "임대차계약증서", "집문서": "임대차계약증서", "종이": "임대차계약증서",

    # 2. 보증금 및 금전 (보증금_대항력, 임대료_증감)
    "전세금": "임차보증금", "보증금": "임차보증금", "맡긴돈": "임차보증금", 
    "떼인돈": "임차보증금",
    "월세": "차임", "방세": "차임", "다달이내는지출": "차임", 
    "렌트비": "차임", "임대료": "차임",
    "복비": "중개보수", "수수료": "중개보수", "중개비": "중개보수",
    "월세올리기": "차임증액", "인상": "증액", "더달라고함": "증액", 
    "5프로": "5퍼센트상한",
    "월세깎기": "차임감액", "할인": "감액", "내리기": "감액",
    "돈먼저받기": "우선변제권", "순위": "우선변제권", "안전장치": "대항력", 
    "돌려받기": "보증금반환",
    "보험": "반환보증", "허그": "HUG", "나라보증": "보증보험",

    # 3. 계약 상태 및 변화 (계약갱신, 계약해지_명도)
    "연장하기": "계약갱신요구권", "한번더살기": "계약갱신", 
    "2플러스2": "계약갱신요구권", "갱신": "계약갱신",
    "재계약": "계약갱신", "자동연장": "묵시적갱신", "연락없음": "묵시적갱신", 
    "그냥연장": "묵시적갱신",
    "이사": "주택의인도", "짐빼기": "주택의인도", "퇴거": "주택의인도", 
    "방빼": "계약해지",
    "주소옮기기": "주민등록", "전입신고": "주민등록", "주소지이전": "주민등록",
    "집주인바뀜": "임대인지위승계", "주인바뀜": "임대인지위승계", 
    "매매": "임대인지위승계",
    "나가라고함": "계약갱신거절", "쫓겨남": "명도", "비워달라": "명도", 
    "중도해지": "계약해지",

    # 4. 수리 및 생활환경 (수선_원상회복, 생활환경_특약)
    "집고치기": "수선의무", "수리": "수선의무", "고쳐줘": "수선의무", 
    "안고쳐줌": "수선의무위반",
    "곰팡이": "하자", "물샘": "누수", "보일러고장": "하자", "파손": "훼손",
    "깨끗이치우기": "원상회복의무", "원래대로해놓기": "원상회복", 
    "청소비": "원상회복비용", "청소": "원상회복",
    "층간소음": "공동생활평온", "옆집소음": "방음", "개키우기": "반려동물특약", 
    "담배": "흡연금지특약",

    # 5. 리스크 및 분쟁 (권리_정보리스크, 분쟁해결)
    "깡통전세": "전세피해", "사기": "전세사기", "경매넘어감": "권리리스크", 
    "빚": "근저당",
    "세금안냄": "체납", "나라빚": "조세채권", "빌린돈": "가압류", 
    "신탁": "신탁부동산",
    "특약": "특약사항", "불공정": "강행규정위반", "독소조항": "불리한약정", 
    "효력있나": "무효여부",
    "조정위": "주택임대차분쟁조정위원회", "소송말고": "분쟁조정", 
    "법원가기싫음": "분쟁조정",
    "집주인사망": "임차권승계", "자식상속": "임차권승계"
}

# 시스템 프롬프트 (답변 생성용)
SYSTEM_PROMPT: str = """
당신은 대한민국 '주택 전월세 사기 예방 및 임대차 법률 전문가 AI'입니다.
사용자의 질문에 대해 제공된 [법적 위계가 정리된 참고 문서]를 바탕으로 답변하세요.

[답변 생성 원칙]
1. **법적 위계 준수**: 
   - 반드시 [SECTION 1: 핵심 법령]의 내용을 최우선 판단 기준으로 삼으세요.
   - [SECTION 1]의 내용이 모호할 때만 [SECTION 2]와 [SECTION 3]를 보충 근거로 활용하세요.
   - 만약 [SECTION 3: 판례]가 [SECTION 1: 법령]과 다르게 해석되는 특수한 경우라면, 
     "원칙은 법령에 따르나, 판례는 예외적으로..."라고 설명하세요.

2. **답변 구조**:
   - **핵심 결론**: 질문에 대한 결론(가능/불가능/유효/무효)을 두괄식으로 요약.
   - **법적 근거**: "주택임대차보호법 제O조에 따르면..." (SECTION 1 인용)
   - **실무 절차**: 필요시 신고 방법, 서류 등 안내 (SECTION 2 인용)
   - **참고 사례**: 유사한 상황에서의 판결이나 해석 (SECTION 3 인용)
   
3. **주의사항**:
   - 사용자의 계약서 내용이 법령(강행규정)에 위반되면 "효력이 없다(무효)"고 명확히 경고하세요.
   - 법률적 조언일 뿐이므로, 최종적으로는 변호사 등의 전문가 확인이 필요함을 고지하세요.

[법적 위계가 정리된 참고 문서]
{context}
"""

# 질문 표준화 프롬프트
NORMALIZATION_PROMPT: str = """
당신은 법률 AI 챗봇의 전처리 담당자입니다. 
아래 [용어 사전]을 엄격히 준수하여 사용자의 질문을 '법률 표준어'로 변환해 주세요.

[수행 지침]
1. 사전에 있는 단어는 반드시 매핑된 법률 용어로 변경하세요.
2. 단어를 변경할 때 문맥에 맞게 조사(이/가, 을/를 등)나 서술어를 자연스럽게 수정하세요.
3. 사용자의 질문 의도를 왜곡하거나 추가적인 답변을 생성하지 마세요.
4. 오직 '변경된 질문' 텍스트만 출력하세요. (설명 금지)

[용어 사전]
{dictionary}

사용자 질문: {question}
변경된 질문:"""


# ==========================================
# 1. 설정 클래스 (Dataclass)
# ==========================================

@dataclass
class RAGConfig:
    """
    RAG 파이프라인의 모든 설정을 중앙에서 관리하는 설정 클래스.
    
    Attributes:
        llm_model: 사용할 LLM 모델명 (기본: exaone3.5:2.4b)
        temperature: LLM temperature (기본: 0.1)
        normalize_temperature: 전처리 LLM temperature (기본: 0)
        embedding_model: 임베딩 모델명
        k_law: Law 인덱스에서 검색할 문서 수
        k_rule: Rule 인덱스에서 검색할 문서 수
        k_case: Case 인덱스에서 검색할 문서 수
        search_multiplier: 초기 검색 시 배수 (Rerank 전)
        rerank_threshold: Rerank 관련도 점수 임계값
        enable_rerank: Reranking 활성화 여부
        rerank_model: Cohere Rerank 모델명
    """
    # LLM 설정
    llm_model: str = "exaone3.5:2.4b"
    temperature: float = 0.1
    normalize_temperature: float = 0.0
    
    # 임베딩 설정
    embedding_model: str = "solar-embedding-1-large-passage"
    
    # 검색 설정
    k_law: int = 5
    k_rule: int = 5
    k_case: int = 3
    search_multiplier: int = 2  # from ge.py: 초기 검색 시 k * multiplier
    
    # Reranking 설정
    enable_rerank: bool = True
    rerank_threshold: float = 0.2
    rerank_model: str = "rerank-multilingual-v3.0"
    
    # 판례 검색 설정
    case_context_top_k: int = 50
    
    def __post_init__(self):
        """설정 유효성 검사"""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature는 0~2 사이여야 합니다.")
        if self.rerank_threshold < 0 or self.rerank_threshold > 1:
            raise ValueError("rerank_threshold는 0~1 사이여야 합니다.")


# ==========================================
# 2. RAG 파이프라인 클래스
# ==========================================

class RAGPipeline:
    """
    주택임대차 RAG 시스템의 메인 파이프라인 클래스.
    
    이 클래스는 VectorStore 초기화, 질문 표준화, 검색, 답변 생성을 
    하나의 인터페이스로 제공합니다.
    
    Usage:
        # 기본 사용
        pipeline = RAGPipeline()
        answer = pipeline.generate_answer("집주인이 월세를 올려달라고 해요")
        
        # 커스텀 설정 사용
        config = RAGConfig(llm_model="exaone3.5:7.8b", k_law=7)
        pipeline = RAGPipeline(config)
    """
    
    def __init__(
        self, 
        config: Optional[RAGConfig] = None,
        pc_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None
    ):
        """
        RAG 파이프라인을 초기화합니다.
        
        Args:
            config: RAGConfig 객체 (없으면 기본값 사용)
            pc_api_key: Pinecone API 키 (없으면 환경변수에서 로드)
            cohere_api_key: Cohere API 키 (없으면 환경변수에서 로드)
        """
        # 환경 변수 로드
        load_dotenv(override=True)
        
        # 설정 초기화
        self.config = config or RAGConfig()
        
        # API 키 설정
        self._pc_api_key = pc_api_key or os.getenv("PINECONE_API_KEY")
        self._cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        
        if not self._pc_api_key:
            raise ValueError("PINECONE_API_KEY가 필요합니다.")
        
        # VectorStore 초기화
        self._law_store: Optional[PineconeVectorStore] = None
        self._rule_store: Optional[PineconeVectorStore] = None
        self._case_store: Optional[PineconeVectorStore] = None
        
        # LLM 인스턴스 (재사용을 위해 캐싱)
        self._normalize_llm: Optional[ChatOllama] = None
        self._generation_llm: Optional[ChatOllama] = None
        
        # Cohere 클라이언트
        self._cohere_client: Optional[Any] = None
        
        # 초기화 실행
        self._initialize()
    
    def _initialize(self) -> None:
        """내부 초기화: VectorStore 및 LLM 인스턴스 생성"""
        # 임베딩 초기화
        embedding = UpstageEmbeddings(model=self.config.embedding_model)
        
        logger.info("🔗 Pinecone 3중 인덱스 연결 중...")
        
        # VectorStore 초기화
        for key, index_name in INDEX_NAMES.items():
            store = PineconeVectorStore(
                index_name=index_name,
                embedding=embedding,
                pinecone_api_key=self._pc_api_key
            )
            setattr(self, f"_{key}_store", store)
        
        logger.info("✅ [Law / Rule / Case] 3개 인덱스 로드 완료!")
        
        # LLM 인스턴스 생성 (재사용)
        self._normalize_llm = ChatOllama(
            model=self.config.llm_model, 
            temperature=self.config.normalize_temperature
        )
        self._generation_llm = ChatOllama(
            model=self.config.llm_model, 
            temperature=self.config.temperature
        )
        
        # Cohere 클라이언트 초기화 (선택적)
        if self.config.enable_rerank and COHERE_AVAILABLE and self._cohere_api_key:
            self._cohere_client = cohere.Client(api_key=self._cohere_api_key)
            logger.info("✅ Cohere Reranking 활성화")
        elif self.config.enable_rerank:
            logger.warning("⚠️ Cohere를 사용할 수 없습니다. Reranking이 비활성화됩니다.")
            self.config.enable_rerank = False
    
    # ==========================================
    # 속성 (Properties)
    # ==========================================
    
    @property
    def law_store(self) -> PineconeVectorStore:
        """Law VectorStore 반환"""
        if self._law_store is None:
            raise RuntimeError("VectorStore가 초기화되지 않았습니다.")
        return self._law_store
    
    @property
    def rule_store(self) -> PineconeVectorStore:
        """Rule VectorStore 반환"""
        if self._rule_store is None:
            raise RuntimeError("VectorStore가 초기화되지 않았습니다.")
        return self._rule_store
    
    @property
    def case_store(self) -> PineconeVectorStore:
        """Case VectorStore 반환"""
        if self._case_store is None:
            raise RuntimeError("VectorStore가 초기화되지 않았습니다.")
        return self._case_store
    
    # ==========================================
    # 핵심 기능 메서드
    # ==========================================
    
    def normalize_query(self, user_query: str) -> str:
        """
        사용자 질문을 법률 용어로 표준화합니다.
        
        Args:
            user_query: 사용자의 원본 질문
            
        Returns:
            표준화된 질문 문자열
        """
        prompt = ChatPromptTemplate.from_template(NORMALIZATION_PROMPT)
        chain = prompt | self._normalize_llm | StrOutputParser()
        
        try:
            normalized = chain.invoke({
                "dictionary": KEYWORD_DICT,
                "question": user_query
            })
            return normalized.strip()
        except Exception as e:
            logger.warning(f"⚠️ 전처리 실패 (원본 사용): {e}")
            return user_query
    
    def get_full_case_context(self, case_no: str) -> str:
        """
        특정 사건번호의 판례 전문을 가져옵니다.
        
        Args:
            case_no: 사건번호 (예: "2020나56247")
            
        Returns:
            판례 전문 텍스트
        """
        try:
            results = self.case_store.similarity_search(
                query="판례 전문 검색",  # API 요구사항을 위한 더미 쿼리
                k=self.config.case_context_top_k,
                filter={"case_no": {"$eq": case_no}}
            )
            
            # chunk_id 순 정렬
            sorted_docs = sorted(
                results, 
                key=lambda x: x.metadata.get('chunk_id', '')
            )
            
            # 중복 제거
            seen_chunks: set = set()
            unique_docs: List[Document] = []
            for doc in sorted_docs:
                cid = doc.metadata.get('chunk_id')
                if cid and cid not in seen_chunks:
                    unique_docs.append(doc)
                    seen_chunks.add(cid)
            
            return "\n".join([doc.page_content for doc in unique_docs])
            
        except Exception as e:
            logger.warning(f"⚠️ 판례 전문 로딩 실패 ({case_no}): {e}")
            return ""
    
    def triple_hybrid_retrieval(self, query: str) -> List[Document]:
        """
        Law, Rule, Case 인덱스에서 문서를 검색하고 Reranking을 수행합니다.
        
        Args:
            query: 검색 쿼리 (표준화된 질문 권장)
            
        Returns:
            법적 위계 순으로 정렬된 Document 리스트
        """
        logger.info(f"🔍 [통합 검색] 쿼리: '{query}'")
        
        cfg = self.config
        multiplier = cfg.search_multiplier
        
        # 1. 병렬 검색 (Parallel Retrieval) - from ge.py: ×2 배수로 넉넉히 검색
        docs_law = self.law_store.similarity_search(
            query, k=cfg.k_law * multiplier
        )
        docs_rule = self.rule_store.similarity_search(
            query, k=cfg.k_rule * multiplier
        )
        docs_case_initial = self.case_store.similarity_search(
            query, k=cfg.k_case * multiplier
        )
        
        # 2. 판례 문맥 확장 (Context Expansion)
        docs_case_expanded: List[Document] = []
        seen_cases: set = set()
        
        for doc in docs_case_initial:
            case_no = doc.metadata.get('case_no')
            if case_no and case_no not in seen_cases:
                full_text = self.get_full_case_context(case_no)
                if full_text:
                    # 판례 전문으로 교체 (메타데이터 유지)
                    doc.page_content = (
                        f"[판례 전문: {doc.metadata.get('title')}]\n{full_text}"
                    )
                    docs_case_expanded.append(doc)
                    seen_cases.add(case_no)
                
                if len(docs_case_expanded) >= cfg.k_case:
                    break
        
        # 3. 문서 통합 (Law + Rule + Case)
        combined_docs = docs_law + docs_rule + docs_case_expanded
        
        # 4. Reranking (선택적)
        selected_docs = combined_docs
        
        if cfg.enable_rerank and self._cohere_client:
            try:
                docs_content = [d.page_content for d in combined_docs]
                rerank_results = self._cohere_client.rerank(
                    model=cfg.rerank_model,
                    query=query,
                    documents=docs_content,
                    top_n=len(combined_docs)
                )
                
                filtered_docs: List[Document] = []
                logger.info(
                    f"📊 Rerank 결과 (총 {len(combined_docs)}개, "
                    f"Threshold: {cfg.rerank_threshold}):"
                )
                
                for r in rerank_results.results:
                    if r.relevance_score > cfg.rerank_threshold:
                        doc = combined_docs[r.index]
                        p = doc.metadata.get('priority', 99)
                        t = doc.metadata.get('title', 'Untitled')
                        logger.info(
                            f" - [Score: {r.relevance_score:.4f}] [P-{p}] {t}"
                        )
                        filtered_docs.append(doc)
                
                selected_docs = filtered_docs
                
            except Exception as e:
                logger.warning(f"⚠️ Rerank 실패 (기본 병합 반환): {e}")
        
        # 5. Priority Sorting (법적 위계 정렬)
        sorted_docs = sorted(
            selected_docs, 
            key=lambda x: int(x.metadata.get('priority', 99))
        )
        
        return sorted_docs
    
    @staticmethod
    def format_context_with_hierarchy(docs: List[Document]) -> str:
        """
        검색된 문서를 법적 위계(Priority)에 따라 섹션별로 재구성합니다.
        
        Args:
            docs: Document 리스트
            
        Returns:
            위계 구조화된 컨텍스트 문자열
        """
        section_1_law: List[str] = []   # Priority 1, 2, 4, 5 (법률, 시행령)
        section_2_rule: List[str] = []  # Priority 3, 6, 7, 8, 11 (규칙, 조례)
        section_3_case: List[str] = []  # Priority 9 (판례, 해석)
        
        for doc in docs:
            p = int(doc.metadata.get('priority', 99))
            src = doc.metadata.get('src_title', '자료')
            title = doc.metadata.get('title', '')
            content = doc.page_content
            
            entry = f"[{src}] {title}\n{content}"
            
            if p in [1, 2, 4, 5]:
                section_1_law.append(entry)
            elif p in [3, 6, 7, 8, 11]:
                section_2_rule.append(entry)
            else:
                section_3_case.append(entry)
        
        # 최종 컨텍스트 조립
        formatted_text = ""
        
        if section_1_law:
            formatted_text += (
                "## [SECTION 1: 핵심 법령 (최우선 법적 근거)]\n" 
                + "\n\n".join(section_1_law) + "\n\n"
            )
        if section_2_rule:
            formatted_text += (
                "## [SECTION 2: 관련 규정 및 절차 (세부 기준)]\n" 
                + "\n\n".join(section_2_rule) + "\n\n"
            )
        if section_3_case:
            formatted_text += (
                "## [SECTION 3: 판례 및 해석 사례 (적용 예시)]\n" 
                + "\n\n".join(section_3_case) + "\n\n"
            )
        
        return formatted_text
    
    def generate_answer(
        self, 
        user_input: str,
        skip_normalization: bool = False
    ) -> str:
        """
        사용자 질문에 대한 최종 답변을 생성합니다.
        
        Args:
            user_input: 사용자의 원본 질문
            skip_normalization: True면 질문 표준화 과정을 건너뜀
            
        Returns:
            최종 답변 문자열
        """
        # 1. 질문 표준화
        if skip_normalization:
            normalized_query = user_input
        else:
            normalized_query = self.normalize_query(user_input)
            logger.info(f"🔄 표준화된 질문: {normalized_query}")
        
        # 2. 통합 검색 및 위계 정렬
        retrieved_docs = self.triple_hybrid_retrieval(normalized_query)
        
        if not retrieved_docs:
            return "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다."
        
        # 3. 위계 구조화된 컨텍스트 생성
        hierarchical_context = self.format_context_with_hierarchy(retrieved_docs)
        
        # 4. LLM 답변 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        
        chain = prompt | self._generation_llm | StrOutputParser()
        
        logger.info("🤖 답변 생성 중...")
        return chain.invoke({
            "context": hierarchical_context, 
            "question": normalized_query
        })


# ==========================================
# 3. 레거시 호환 함수 (Backward Compatibility)
# ==========================================

# 전역 파이프라인 인스턴스 (레거시 코드 지원용)
_global_pipeline: Optional[RAGPipeline] = None


def initialize_vector_stores(
    pc_api_key: Optional[str] = None,
    up_api_key: Optional[str] = None
) -> Tuple[PineconeVectorStore, PineconeVectorStore, PineconeVectorStore]:
    """
    레거시 호환 함수: VectorStore 3개를 초기화하여 반환합니다.
    
    새 코드에서는 RAGPipeline 클래스를 직접 사용하는 것을 권장합니다.
    
    Returns:
        (law_store, rule_store, case_store) 튜플
    """
    global _global_pipeline
    _global_pipeline = RAGPipeline(pc_api_key=pc_api_key)
    return (
        _global_pipeline.law_store,
        _global_pipeline.rule_store,
        _global_pipeline.case_store
    )


def normalize_query(user_query: str, llm_model: str = "exaone3.5:2.4b") -> str:
    """
    레거시 호환 함수: 사용자 쿼리를 법률 용어로 표준화합니다.
    
    새 코드에서는 RAGPipeline.normalize_query()를 사용하세요.
    """
    if _global_pipeline:
        return _global_pipeline.normalize_query(user_query)
    
    # 파이프라인 없이 단독 실행
    llm = ChatOllama(model=llm_model, temperature=0)
    prompt = ChatPromptTemplate.from_template(NORMALIZATION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    try:
        return chain.invoke({
            "dictionary": KEYWORD_DICT,
            "question": user_query
        }).strip()
    except Exception as e:
        logger.warning(f"⚠️ 전처리 실패: {e}")
        return user_query


def generate_final_answer(
    user_input: str,
    law_store: PineconeVectorStore,
    rule_store: PineconeVectorStore,
    case_store: PineconeVectorStore,
    llm_model: str = "exaone3.5:2.4b",
    temperature: float = 0.1,
    k_law: int = 3,
    k_rule: int = 3,
    k_case: int = 2,
    score_threshold: float = 0.2
) -> str:
    """
    레거시 호환 함수: 사용자 질문에 대한 최종 답변을 생성합니다.
    
    새 코드에서는 RAGPipeline.generate_answer()를 사용하세요.
    """
    if _global_pipeline:
        return _global_pipeline.generate_answer(user_input)
    
    # 파이프라인 없이 단독 실행 (임시 인스턴스 생성)
    config = RAGConfig(
        llm_model=llm_model,
        temperature=temperature,
        k_law=k_law,
        k_rule=k_rule,
        k_case=k_case,
        rerank_threshold=score_threshold
    )
    pipeline = RAGPipeline(config)
    return pipeline.generate_answer(user_input)


# ==========================================
# 4. Django/FastAPI 통합 가이드
# ==========================================

"""
=== Django 통합 예시 ===

# settings.py
RAG_CONFIG = {
    'llm_model': 'exaone3.5:2.4b',
    'temperature': 0.1,
    'k_law': 5,
    'k_rule': 5,
    'k_case': 3,
}

# apps.py
from django.apps import AppConfig
from rag_module_unified import RAGPipeline, RAGConfig

class ChatbotConfig(AppConfig):
    name = 'chatbot'
    pipeline = None
    
    def ready(self):
        from django.conf import settings
        config = RAGConfig(**settings.RAG_CONFIG)
        ChatbotConfig.pipeline = RAGPipeline(config)

# views.py
from django.http import JsonResponse
from .apps import ChatbotConfig

def chat_view(request):
    if request.method == 'POST':
        question = request.POST.get('question', '')
        answer = ChatbotConfig.pipeline.generate_answer(question)
        return JsonResponse({'answer': answer})


=== FastAPI 통합 예시 ===

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_module_unified import RAGPipeline, RAGConfig
from contextlib import asynccontextmanager

# 전역 파이프라인
pipeline: RAGPipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    config = RAGConfig(llm_model="exaone3.5:2.4b")
    pipeline = RAGPipeline(config)
    yield
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)

class Question(BaseModel):
    text: str

@app.post("/chat")
async def chat(question: Question):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    answer = pipeline.generate_answer(question.text)
    return {"answer": answer}
"""


# ==========================================
# 5. 테스트 실행 블록
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 RAG LLM Pipeline (Unified) 테스트 시작")
    print("=" * 70)
    
    try:
        # 방법 1: 클래스 기반 (권장)
        print("\n[방법 1] RAGPipeline 클래스 사용")
        print("-" * 50)
        
        # 커스텀 설정으로 파이프라인 생성
        config = RAGConfig(
            llm_model="exaone3.5:2.4b",
            temperature=0.1,
            k_law=3,
            k_rule=3,
            k_case=2
        )
        pipeline = RAGPipeline(config)
        
        # 테스트 쿼리
        test_queries = [
            "집주인이 월세를 올려 달래요. 거절하니까 나가라고 하는데 어떡하죠?",
            "전입신고는 언제까지 해야 하나요?",
            "집주인이 실거주한다고 나가라고 하는데, 진짜인지 의심스러워요."
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 테스트 {i}: {query}")
            print("-" * 50)
            answer = pipeline.generate_answer(query)
            print(answer)
            print("=" * 70)
        
        # 방법 2: 레거시 함수 사용 (이전 코드 호환)
        print("\n[방법 2] 레거시 함수 사용 (Backward Compatibility)")
        print("-" * 50)
        
        law, rule, case = initialize_vector_stores()
        answer = generate_final_answer(
            "보증금 돌려받으려면 어떻게 해야 하나요?",
            law, rule, case
        )
        print(answer)
        
    except Exception as e:
        logger.error(f"🔥 에러 발생: {e}")
        raise
