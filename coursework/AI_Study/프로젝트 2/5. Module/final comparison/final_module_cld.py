"""
Unified RAG Module - Final Version
주택임대차 RAG 시스템 - 하이브리드 검색 및 답변 생성 모듈

[융합 출처]
- improved_module_cg.py: 자체 BM25 구현, 상세 설정, embedding 백엔드 선택
- improved_module_cl.py: 모듈화된 클래스 구조 (Tokenizer, BM25Scorer, ScoreFusion)
- improved_module_ge.py: BM25Retriever 옵션, 간결한 파이프라인

[LLM 설정]
- normalize_query: Upstage Solar-Pro2 (ChatUpstage)
- generate_answer: OpenAI GPT-4o-mini (ChatOpenAI)

[핵심 기능]
1. Dense 검색: Pinecone + Solar Embedding
2. Sparse 검색: BM25 (자체 구현 또는 rank_bm25)
3. Hybrid Fusion: RRF / Weighted / Rank-Sum
4. Rerank: Cohere rerank-multilingual-v3.0
5. 2-Stage Case Expansion
6. 법적 위계(Priority) 기반 컨텍스트 구성

[의존성]
pip install langchain-upstage langchain-openai langchain-pinecone cohere
pip install rank-bm25 kiwipiepy  # 선택적
"""

from __future__ import annotations

import logging
import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, Iterable, List, 
    Optional, Sequence, Tuple, Union
)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Embedding
from langchain_upstage import UpstageEmbeddings

# LLM backends
try:
    from langchain_upstage import ChatUpstage
    UPSTAGE_CHAT_AVAILABLE = True
except ImportError:
    ChatUpstage = None
    UPSTAGE_CHAT_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    ChatOpenAI = None
    OPENAI_AVAILABLE = False

# Fallback LLM (Ollama)
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        OLLAMA_AVAILABLE = True
    except ImportError:
        ChatOllama = None
        OLLAMA_AVAILABLE = False

# Vector Store
from langchain_pinecone import PineconeVectorStore

# BM25 (optional - use built-in if not available)
try:
    from rank_bm25 import BM25Okapi, BM25Plus
    RANK_BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25Plus = None
    RANK_BM25_AVAILABLE = False

# Korean Tokenizer (optional)
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    Kiwi = None
    KIWI_AVAILABLE = False

# Cohere Rerank (optional)
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    cohere = None
    COHERE_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
INDEX_NAMES: Dict[str, str] = {
    "law": "law-index-final",
    "rule": "rule-index-final",
    "case": "case-index-final",
}

# 법률 용어 사전 (질문 표준화용) - 통합 버전
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

    # 2. 보증금 및 금전
    "보증금": "임대차보증금", "전세금": "임대차보증금", "보증보험": "보증금반환보증",
    "돈못받음": "보증금미반환", "안돌려줌": "보증금미반환", "못돌려받음": "보증금미반환",
    "월세": "차임", "관리비": "관리비", "연체": "차임연체", "밀림": "차임연체",
    "복비": "중개보수", "수수료": "중개보수", "중개비": "중개보수",
    "월세올리기": "차임증액", "인상": "증액", "더달라고함": "증액",
    "월세깎기": "차임감액", "할인": "감액", "내리기": "감액",
    "돈먼저받기": "우선변제권", "순위": "우선변제권", "안전장치": "대항력",
    "돌려받기": "보증금반환",

    # 3. 기간 및 종료/갱신
    "재계약": "계약갱신", "연장": "계약갱신", "갱신": "계약갱신",
    "갱신청구": "계약갱신요구권", "2년더": "계약갱신요구권", "2플러스2": "계약갱신요구권",
    "자동연장": "묵시적갱신", "묵시": "묵시적갱신", "연락없음": "묵시적갱신",
    "이사": "주택의인도", "짐빼기": "주택의인도", "퇴거": "주택의인도",
    "방빼": "계약해지", "중도해지": "계약해지",
    "주소옮기기": "주민등록", "전입신고": "주민등록", "주소지이전": "주민등록",
    "집주인바뀜": "임대인지위승계", "주인바뀜": "임대인지위승계",
    "매매": "임대인지위승계",
    "나가라고함": "계약갱신거절", "쫓겨남": "명도", "비워달라": "명도",

    # 4. 수리 및 생활환경
    "집고치기": "수선의무", "수리": "수선의무", "고쳐줘": "수선의무",
    "안고쳐줌": "수선의무위반",
    "곰팡이": "하자", "물샘": "누수", "보일러고장": "하자", "파손": "훼손",
    "깨끗이치우기": "원상회복의무", "원래대로해놓기": "원상회복",
    "청소비": "원상회복비용", "청소": "원상회복",
    "층간소음": "공동생활평온", "옆집소음": "방음", "개키우기": "반려동물특약",
    "담배": "흡연금지특약",

    # 5. 권리/대항력/확정일자
    "확정일자": "확정일자", "전입": "주민등록", "대항력": "대항력",
    "우선변제": "우선변제권", "최우선": "최우선변제권",
    "경매": "경매절차", "공매": "공매절차",
    "등기": "등기부등본", "등본": "등기부등본",
    "근저당": "근저당권", "가압류": "가압류", "가처분": "가처분",
    "깡통전세": "전세피해", "사기": "전세사기", "경매넘어감": "권리리스크",

    # 6. 분쟁 해결
    "내용증명": "내용증명", "소송": "소송", "민사": "민사소송",
    "조정위": "주택임대차분쟁조정위원회", "소송말고": "분쟁조정",
    "법원가기싫음": "분쟁조정",
    "집주인사망": "임차권승계", "자식상속": "임차권승계",
    "특약": "특약사항", "불공정": "강행규정위반", "독소조항": "불리한약정",
    "효력있나": "무효여부",
}

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
NORMALIZATION_PROMPT: str = """당신은 법률 AI 챗봇의 전처리 담당자입니다.
아래 [용어 사전]을 엄격히 준수하여 사용자의 질문을 '법률 표준어'로 변환해 주세요.

[수행 지침]
1. 사전에 있는 단어는 반드시 매핑된 법률 용어로 변경하세요.
2. 단어를 변경할 때 문맥에 맞게 조사(이/가, 을/를 등)나 서술어를 자연스럽게 수정하세요.
3. 사용자의 질문 의도를 왜곡하거나 추가적인 답변, 별도의 설명을 생성하지 마세요.
4. 변경 전 단어 뒤에 변경된 단어를 괄호로 덧붙여 텍스트만 출력하세요. ex. "집주인(임대인)이..."

[용어 사전]
{dictionary}

사용자 질문: {question}
변경된 질문:"""

SYSTEM_PROMPT: str = """당신은 대한민국 '주택 전월세 사기 예방 및 임대차 법률 전문가 AI'입니다.
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
{context}"""

# --------------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------------
def _safe_int(x: object, default: int = 99) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _dedupe_docs(
    docs: Iterable[Document],
    key_fields: Sequence[str] = ("chunk_id", "id"),
) -> List[Document]:
    """메타데이터 기반 중복 제거"""
    seen: set = set()
    out: List[Document] = []
    for d in docs:
        md = d.metadata or {}
        key = None
        for f in key_fields:
            v = md.get(f)
            if v:
                key = f"{f}:{v}"
                break
        if key is None:
            key = f"content:{hash(d.page_content)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# --------------------------------------------------------------------------------------
# Tokenizer Classes (from improved_module_cl.py)
# --------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


class Tokenizer(ABC):
    """토크나이저 추상 클래스"""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


class SimpleTokenizer(Tokenizer):
    """정규식 기반 단순 토크나이저"""
    
    def __init__(self, min_length: int = 1):
        self.min_length = min_length
    
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = _TOKEN_RE.findall(text.lower())
        return [t for t in tokens if len(t) >= self.min_length]


class KiwiTokenizer(Tokenizer):
    """Kiwi 기반 한국어 형태소 분석 토크나이저"""
    
    def __init__(
        self,
        pos_tags: Optional[Tuple[str, ...]] = None,
        min_length: int = 1
    ):
        if not KIWI_AVAILABLE:
            raise ImportError("kiwipiepy가 설치되지 않았습니다: pip install kiwipiepy")
        
        self._kiwi = Kiwi()
        self.pos_tags = pos_tags or ('NNG', 'NNP', 'VV', 'VA', 'SL', 'SH')
        self.min_length = min_length
    
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = []
        for token in self._kiwi.tokenize(text):
            if token.tag in self.pos_tags and len(token.form) >= self.min_length:
                tokens.append(token.form.lower())
        return tokens


def get_default_tokenizer() -> Tokenizer:
    """사용 가능한 최적의 토크나이저 반환"""
    if KIWI_AVAILABLE:
        return KiwiTokenizer()
    return SimpleTokenizer()


# --------------------------------------------------------------------------------------
# BM25 Implementation (from improved_module_cg.py - no external dependency)
# --------------------------------------------------------------------------------------
def _bm25_scores_builtin(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    Built-in BM25Okapi implementation (no external dependency)
    """
    N = len(docs_tokens)
    if N == 0:
        return []
    if not query_tokens:
        return [0.0] * N

    doc_lens = [len(toks) for toks in docs_tokens]
    avgdl = sum(doc_lens) / N if N else 1.0
    if avgdl <= 0:
        avgdl = 1.0

    # Document frequency
    df: Dict[str, int] = defaultdict(int)
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1

    # IDF
    idf: Dict[str, float] = {}
    for t, dfi in df.items():
        idf[t] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

    # Query term frequency
    qtf = Counter(query_tokens)

    scores: List[float] = []
    for toks, dl in zip(docs_tokens, doc_lens):
        tf = Counter(toks)
        score = 0.0
        norm = (1.0 - b) + b * (dl / avgdl)
        for term, qf in qtf.items():
            if term not in tf:
                continue
            f = tf[term]
            denom = f + k1 * norm
            if denom == 0:
                continue
            score += (idf.get(term, 0.0) * (f * (k1 + 1.0) / denom)) * (1.0 + 0.1 * (qf - 1))
        scores.append(float(score))
    return scores


# --------------------------------------------------------------------------------------
# BM25 Scorer Class (from improved_module_cl.py)
# --------------------------------------------------------------------------------------
class BM25Scorer:
    """BM25 기반 문서 스코어링"""
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        algorithm: str = "okapi",  # "okapi", "plus", or "builtin"
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.algorithm = algorithm
        self.k1 = k1
        self.b = b
        
        self._bm25: Optional[Any] = None
        self._corpus_tokens: List[List[str]] = []
        self._use_builtin = (
            algorithm == "builtin" or 
            not RANK_BM25_AVAILABLE
        )
    
    def fit(self, documents: List[Document]) -> "BM25Scorer":
        """문서 코퍼스로 BM25 인덱스 구축"""
        self._corpus_tokens = [
            self.tokenizer.tokenize(doc.page_content or "")
            for doc in documents
        ]
        
        if not self._use_builtin and RANK_BM25_AVAILABLE:
            BM25Class = BM25Plus if self.algorithm == "plus" else BM25Okapi
            self._bm25 = BM25Class(self._corpus_tokens, k1=self.k1, b=self.b)
        
        return self
    
    def score(self, query: str) -> List[float]:
        """쿼리에 대한 각 문서의 BM25 점수 반환"""
        query_tokens = self.tokenizer.tokenize(query)
        
        if self._use_builtin or self._bm25 is None:
            return _bm25_scores_builtin(
                query_tokens, 
                self._corpus_tokens,
                k1=self.k1,
                b=self.b
            )
        
        return self._bm25.get_scores(query_tokens).tolist()


# --------------------------------------------------------------------------------------
# Score Fusion (from improved_module_cl.py + improved_module_cg.py)
# --------------------------------------------------------------------------------------
class ScoreFusion:
    """Dense와 Sparse 점수를 결합하는 전략"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        dense_ranks: Dict[str, int],
        sparse_ranks: Dict[str, int],
        k: int = 60,
        w_dense: float = 1.0,
        w_sparse: float = 1.0,
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion (RRF)"""
        all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        scores: Dict[str, float] = {}
        
        for doc_id in all_docs:
            score = 0.0
            if doc_id in dense_ranks:
                score += w_dense / (k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += w_sparse / (k + sparse_ranks[doc_id])
            scores[doc_id] = score
        
        return scores
    
    @staticmethod
    def weighted_sum(
        dense_scores: Dict[str, float],
        sparse_scores: Dict[str, float],
        alpha: float = 0.5,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """가중 합산: alpha * dense + (1-alpha) * sparse"""
        if normalize:
            dense_scores = ScoreFusion._normalize(dense_scores)
            sparse_scores = ScoreFusion._normalize(sparse_scores)
        
        all_docs = set(dense_scores.keys()) | set(sparse_scores.keys())
        scores: Dict[str, float] = {}
        
        for doc_id in all_docs:
            d_score = dense_scores.get(doc_id, 0.0)
            s_score = sparse_scores.get(doc_id, 0.0)
            scores[doc_id] = alpha * d_score + (1 - alpha) * s_score
        
        return scores
    
    @staticmethod
    def rank_sum(
        dense_ranks: Dict[str, int],
        sparse_ranks: Dict[str, int],
        w_dense: float = 0.6,
        w_sparse: float = 0.4,
    ) -> Dict[str, float]:
        """Rank 기반 합산 (0~1 정규화 후 가중 합)"""
        all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        n = len(all_docs)
        if n <= 1:
            return {doc_id: w_dense + w_sparse for doc_id in all_docs}
        
        def to_unit(r: int) -> float:
            return 1.0 - (r - 1) / (n - 1)
        
        scores: Dict[str, float] = {}
        for doc_id in all_docs:
            d_rank = dense_ranks.get(doc_id, n)
            s_rank = sparse_ranks.get(doc_id, n)
            scores[doc_id] = w_dense * to_unit(d_rank) + w_sparse * to_unit(s_rank)
        
        return scores
    
    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """Min-Max 정규화"""
        if not scores:
            return scores
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class RAGConfig:
    """RAG 파이프라인 설정"""
    
    # ============ LLM Settings ============
    # Normalize: Upstage Solar-Pro2
    normalize_model: str = "solar-pro2"
    normalize_temperature: float = 0.0
    
    # Generate: OpenAI GPT-4o-mini
    generation_model: str = "gpt-4o-mini"
    generation_temperature: float = 0.1
    
    # Fallback LLM (Ollama)
    fallback_model: str = "exaone3.5:2.4b"
    
    # ============ Embedding Settings ============
    embedding_model: str = "solar-embedding-1-large-passage"
    
    # ============ Retrieval Settings ============
    k_law: int = 5
    k_rule: int = 5
    k_case: int = 3
    search_multiplier: int = 2
    
    # ============ Hybrid Search Settings ============
    enable_hybrid: bool = True
    hybrid_method: str = "rrf"  # "rrf", "weighted", "rank_sum"
    hybrid_alpha: float = 0.5   # For weighted method
    rrf_k: int = 60
    hybrid_dense_weight: float = 0.6
    hybrid_sparse_weight: float = 0.4
    
    # ============ BM25 Settings ============
    bm25_algorithm: str = "builtin"  # "builtin", "okapi", "plus"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_max_doc_chars: int = 4000
    use_kiwi_tokenizer: bool = True
    
    # ============ Rerank Settings ============
    enable_rerank: bool = True
    rerank_threshold: float = 0.2
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_max_documents: int = 80
    rerank_doc_max_chars: int = 2000
    
    # ============ Case Expansion Settings ============
    case_candidate_k: int = 40
    case_expand_top_n: Optional[int] = None
    case_context_top_k: int = 50
    
    # ============ Deduplication Settings ============
    dedupe_key_fields: Tuple[str, ...] = ("chunk_id", "id")
    
    def __post_init__(self) -> None:
        if not (0 <= self.generation_temperature <= 2):
            raise ValueError("generation_temperature는 0~2 사이여야 합니다.")
        if not (0 <= self.rerank_threshold <= 1):
            raise ValueError("rerank_threshold는 0~1 사이여야 합니다.")
        if self.hybrid_method not in ("rrf", "weighted", "rank_sum"):
            raise ValueError("hybrid_method는 'rrf', 'weighted', 'rank_sum' 중 하나여야 합니다.")


# --------------------------------------------------------------------------------------
# RAG Pipeline
# --------------------------------------------------------------------------------------
class RAGPipeline:
    """
    Unified RAG Pipeline
    
    - normalize_query: Upstage Solar-Pro2
    - generate_answer: OpenAI GPT-4o-mini
    - Hybrid Search: Dense (Solar Embedding) + Sparse (BM25)
    
    Usage:
        pipeline = RAGPipeline()
        answer = pipeline.generate_answer("집주인이 보증금을 안 돌려줘요")
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        *,
        pc_api_key: Optional[str] = None,
        upstage_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
    ) -> None:
        self.config = config or RAGConfig()
        
        # API Keys
        self._pc_api_key = pc_api_key or os.getenv("PINECONE_API_KEY")
        self._upstage_api_key = upstage_api_key or os.getenv("UPSTAGE_API_KEY")
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        
        if not self._pc_api_key:
            raise ValueError("PINECONE_API_KEY가 필요합니다.")
        
        # Initialize components
        self._init_embedding()
        self._init_vector_stores()
        self._init_llms()
        self._init_tokenizer()
        self._init_cohere()
    
    def _init_embedding(self) -> None:
        """Embedding 초기화"""
        if not self._upstage_api_key:
            raise ValueError("UPSTAGE_API_KEY가 필요합니다 (Embedding용).")
        
        os.environ.setdefault("UPSTAGE_API_KEY", self._upstage_api_key)
        self._embedding = UpstageEmbeddings(model=self.config.embedding_model)
        logger.info("✅ Upstage Embedding 초기화 완료")
    
    def _init_vector_stores(self) -> None:
        """Vector Store 초기화"""
        logger.info("🔗 Pinecone 3중 인덱스 연결 중...")
        
        self._law_store = PineconeVectorStore(
            index_name=INDEX_NAMES["law"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        self._rule_store = PineconeVectorStore(
            index_name=INDEX_NAMES["rule"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        self._case_store = PineconeVectorStore(
            index_name=INDEX_NAMES["case"],
            embedding=self._embedding,
            pinecone_api_key=self._pc_api_key,
        )
        
        logger.info("✅ [Law / Rule / Case] 3개 인덱스 로드 완료!")
    
    def _init_llms(self) -> None:
        """LLM 초기화 - Solar-Pro2 (normalize) + GPT-4o-mini (generate)"""
        cfg = self.config
        
        # 1. Normalize LLM: Upstage Solar-Pro2
        if UPSTAGE_CHAT_AVAILABLE and self._upstage_api_key:
            os.environ.setdefault("UPSTAGE_API_KEY", self._upstage_api_key)
            self._normalize_llm = ChatUpstage(
                model=cfg.normalize_model,
                temperature=cfg.normalize_temperature,
            )
            logger.info(f"✅ Normalize LLM: Upstage {cfg.normalize_model}")
        elif OLLAMA_AVAILABLE:
            logger.warning(f"⚠️ Upstage 사용 불가. Fallback: {cfg.fallback_model}")
            self._normalize_llm = ChatOllama(
                model=cfg.fallback_model,
                temperature=cfg.normalize_temperature,
            )
        else:
            raise ImportError("LLM 백엔드가 없습니다. langchain-upstage 또는 langchain-ollama를 설치하세요.")
        
        # 2. Generation LLM: OpenAI GPT-4o-mini
        if OPENAI_AVAILABLE and self._openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self._openai_api_key)
            self._generation_llm = ChatOpenAI(
                model=cfg.generation_model,
                temperature=cfg.generation_temperature,
            )
            logger.info(f"✅ Generation LLM: OpenAI {cfg.generation_model}")
        elif OLLAMA_AVAILABLE:
            logger.warning(f"⚠️ OpenAI 사용 불가. Fallback: {cfg.fallback_model}")
            self._generation_llm = ChatOllama(
                model=cfg.fallback_model,
                temperature=cfg.generation_temperature,
            )
        else:
            raise ImportError("LLM 백엔드가 없습니다. langchain-openai 또는 langchain-ollama를 설치하세요.")
    
    def _init_tokenizer(self) -> None:
        """BM25용 토크나이저 초기화"""
        self._tokenizer: Optional[Tokenizer] = None
        
        if self.config.enable_hybrid:
            if self.config.use_kiwi_tokenizer and KIWI_AVAILABLE:
                self._tokenizer = KiwiTokenizer()
                logger.info("✅ Kiwi 토크나이저 초기화 완료")
            else:
                self._tokenizer = SimpleTokenizer()
                logger.info("ℹ️ SimpleTokenizer 사용 (공백 기반)")
    
    def _init_cohere(self) -> None:
        """Cohere Rerank 클라이언트 초기화"""
        self._cohere_client: Optional[Any] = None
        
        if self.config.enable_rerank:
            if COHERE_AVAILABLE and self._cohere_api_key:
                self._cohere_client = cohere.Client(self._cohere_api_key)
                logger.info("✅ Cohere Reranking 활성화")
            else:
                logger.warning("⚠️ Cohere 사용 불가. Rerank 비활성화됨.")
    
    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def law_store(self) -> PineconeVectorStore:
        return self._law_store
    
    @property
    def rule_store(self) -> PineconeVectorStore:
        return self._rule_store
    
    @property
    def case_store(self) -> PineconeVectorStore:
        return self._case_store
    
    # ----------------------------
    # Core Methods
    # ----------------------------
    def normalize_query(self, user_query: str) -> str:
        """사용자 질문을 법률 용어로 표준화 (Solar-Pro2 사용)"""
        prompt = ChatPromptTemplate.from_template(NORMALIZATION_PROMPT)
        chain = prompt | self._normalize_llm | StrOutputParser()
        
        try:
            normalized = chain.invoke({
                "dictionary": KEYWORD_DICT,
                "question": user_query
            })
            return str(normalized).strip()
        except Exception as e:
            logger.warning(f"⚠️ 전처리 실패 (원본 사용): {e}")
            return user_query
    
    def get_full_case_context(self, case_no: str) -> str:
        """특정 사건번호의 판례 전문을 가져옴"""
        try:
            results = self.case_store.similarity_search(
                query="판례 전문 검색",
                k=self.config.case_context_top_k,
                filter={"case_no": {"$eq": case_no}},
            )
            sorted_docs = sorted(
                results,
                key=lambda x: str(x.metadata.get("chunk_id", ""))
            )
            unique_docs = _dedupe_docs(sorted_docs, self.config.dedupe_key_fields)
            return "\n".join([d.page_content for d in unique_docs]).strip()
        except Exception as e:
            logger.warning(f"⚠️ 판례 전문 로딩 실패 ({case_no}): {e}")
            return ""
    
    def _attach_source(self, docs: List[Document], source: str) -> List[Document]:
        """검색 출처를 메타데이터에 주입"""
        for d in docs:
            if d.metadata is None:
                d.metadata = {}
            d.metadata["__source_index"] = source
        return docs
    
    def _get_doc_id(self, doc: Document) -> str:
        """문서의 고유 ID 생성"""
        md = doc.metadata or {}
        for field in self.config.dedupe_key_fields:
            if md.get(field):
                return f"{field}:{md[field]}"
        return f"hash:{hash(doc.page_content)}"
    
    def _search_with_dense_rank(
        self, 
        store: PineconeVectorStore, 
        query: str, 
        k: int
    ) -> List[Document]:
        """Dense 검색 후 순위 메타데이터 추가"""
        try:
            pairs = store.similarity_search_with_score(query, k=k)
            docs: List[Document] = []
            for rank, (doc, score) in enumerate(pairs, start=1):
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["__dense_score"] = float(score)
                doc.metadata["__dense_rank"] = int(rank)
                docs.append(doc)
            return docs
        except Exception:
            docs = store.similarity_search(query, k=k)
            for rank, doc in enumerate(docs, start=1):
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["__dense_rank"] = int(rank)
            return docs
    
    def _hybrid_fusion(self, query: str, docs: List[Document]) -> List[Document]:
        """Dense + BM25 하이브리드 융합"""
        cfg = self.config
        
        if not cfg.enable_hybrid or not docs or not self._tokenizer:
            return docs
        
        docs = _dedupe_docs(docs, cfg.dedupe_key_fields)
        if len(docs) <= 1:
            return docs
        
        # Dense ranks
        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        for i, d in enumerate(docs, start=1):
            doc_id = self._get_doc_id(d)
            dense_ranks[doc_id] = d.metadata.get("__dense_rank", i)
            dense_scores[doc_id] = 1.0 / dense_ranks[doc_id]
        
        # BM25 scores
        scorer = BM25Scorer(
            tokenizer=self._tokenizer,
            algorithm=cfg.bm25_algorithm,
            k1=cfg.bm25_k1,
            b=cfg.bm25_b,
        )
        
        # Truncate for BM25
        truncated_docs = []
        for d in docs:
            truncated_doc = Document(
                page_content=_truncate(d.page_content or "", cfg.bm25_max_doc_chars),
                metadata=d.metadata
            )
            truncated_docs.append(truncated_doc)
        
        scorer.fit(truncated_docs)
        bm25_scores_list = scorer.score(query)
        
        bm25_scores: Dict[str, float] = {}
        for d, score in zip(docs, bm25_scores_list):
            bm25_scores[self._get_doc_id(d)] = score
        
        # BM25 ranks
        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        sparse_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(sorted_bm25, start=1)}
        
        # Fusion
        if cfg.hybrid_method == "rrf":
            fused = ScoreFusion.reciprocal_rank_fusion(
                dense_ranks, sparse_ranks,
                k=cfg.rrf_k,
                w_dense=cfg.hybrid_dense_weight,
                w_sparse=cfg.hybrid_sparse_weight,
            )
        elif cfg.hybrid_method == "weighted":
            fused = ScoreFusion.weighted_sum(
                dense_scores, bm25_scores,
                alpha=cfg.hybrid_alpha,
            )
        else:  # rank_sum
            fused = ScoreFusion.rank_sum(
                dense_ranks, sparse_ranks,
                w_dense=cfg.hybrid_dense_weight,
                w_sparse=cfg.hybrid_sparse_weight,
            )
        
        # Reorder by fused score
        doc_map = {self._get_doc_id(d): d for d in docs}
        sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)
        
        reordered = []
        for rank, doc_id in enumerate(sorted_ids, start=1):
            if doc_id in doc_map:
                d = doc_map[doc_id]
                d.metadata["__hybrid_score"] = fused[doc_id]
                d.metadata["__hybrid_rank"] = rank
                reordered.append(d)
        
        return reordered
    
    def _rerank(
        self,
        query: str,
        docs: List[Document]
    ) -> Optional[List[Tuple[int, float]]]:
        """Cohere Rerank 실행"""
        if not self._cohere_client:
            return None
        
        cfg = self.config
        texts = [_truncate(d.page_content or "", cfg.rerank_doc_max_chars) for d in docs]
        
        try:
            rerank_results = self._cohere_client.rerank(
                model=cfg.rerank_model,
                query=query,
                documents=texts,
                top_n=len(texts),
            )
            return [(r.index, float(r.relevance_score)) for r in rerank_results.results]
        except Exception as e:
            logger.warning(f"⚠️ Rerank 실패: {e}")
            return None
    
    def _cap_for_rerank(
        self,
        law: List[Document],
        rule: List[Document],
        case: List[Document]
    ) -> List[Document]:
        """Rerank 입력 문서 수 제한"""
        cfg = self.config
        law = _dedupe_docs(law, cfg.dedupe_key_fields)
        rule = _dedupe_docs(rule, cfg.dedupe_key_fields)
        case = _dedupe_docs(case, cfg.dedupe_key_fields)
        
        base = law + rule
        if len(base) >= cfg.rerank_max_documents:
            return base[:cfg.rerank_max_documents]
        
        remaining = cfg.rerank_max_documents - len(base)
        return base + case[:remaining]
    
    def triple_hybrid_retrieval(self, query: str) -> List[Document]:
        """
        3중 인덱스 하이브리드 검색
        
        1. Dense 검색 (Pinecone + Solar Embedding)
        2. Hybrid Fusion (Dense + BM25)
        3. Rerank (Cohere)
        4. 2-Stage Case Expansion
        5. Priority 정렬
        """
        cfg = self.config
        mult = cfg.search_multiplier
        
        logger.info(f"🔍 [Hybrid 검색] query='{query}'")
        
        # 1) Dense Retrieval
        docs_law = self._attach_source(
            self._search_with_dense_rank(self.law_store, query, k=cfg.k_law * mult),
            "law",
        )
        docs_rule = self._attach_source(
            self._search_with_dense_rank(self.rule_store, query, k=cfg.k_rule * mult),
            "rule",
        )
        docs_case_chunks = self._attach_source(
            self._search_with_dense_rank(self.case_store, query, k=cfg.case_candidate_k),
            "case",
        )
        
        # 2) Hybrid Fusion (per index)
        if cfg.enable_hybrid:
            docs_law = self._hybrid_fusion(query, docs_law)
            docs_rule = self._hybrid_fusion(query, docs_rule)
            docs_case_chunks = self._hybrid_fusion(query, docs_case_chunks)
        
        # 3) Prepare for Rerank
        combined = self._cap_for_rerank(docs_law, docs_rule, docs_case_chunks)
        
        # 4) Rerank
        selected_docs: List[Document]
        ranked = self._rerank(query, combined) if cfg.enable_rerank else None
        
        if ranked:
            filtered = [(i, s) for (i, s) in ranked if s >= cfg.rerank_threshold]
            if not filtered:
                desired = min(cfg.k_law + cfg.k_rule + cfg.k_case, len(ranked))
                filtered = ranked[:desired]
            selected_docs = [combined[i] for (i, _) in filtered]
            logger.info(f"📌 Rerank 완료: {len(selected_docs)}개 선택")
        else:
            selected_docs = combined
        
        # 5) Deduplicate
        selected_docs = _dedupe_docs(selected_docs, cfg.dedupe_key_fields)
        
        # 6) Select top docs per source + Case Expansion
        law_ranked = [d for d in selected_docs if d.metadata.get("__source_index") == "law"]
        rule_ranked = [d for d in selected_docs if d.metadata.get("__source_index") == "rule"]
        case_ranked = [d for d in selected_docs if d.metadata.get("__source_index") == "case"]
        
        final_law = law_ranked[:cfg.k_law]
        final_rule = rule_ranked[:cfg.k_rule]
        
        # Case Expansion
        top_n = cfg.case_expand_top_n or cfg.k_case
        seen_case_no: set = set()
        chosen_cases: List[Document] = []
        
        for d in case_ranked:
            case_no = d.metadata.get("case_no")
            if not case_no or case_no in seen_case_no:
                continue
            seen_case_no.add(case_no)
            chosen_cases.append(d)
            if len(chosen_cases) >= top_n:
                break
        
        expanded_cases: List[Document] = []
        for d in chosen_cases:
            case_no = d.metadata.get("case_no")
            if not case_no:
                continue
            full_text = self.get_full_case_context(str(case_no))
            if not full_text:
                expanded_cases.append(d)
                continue
            
            title = d.metadata.get("title") or d.metadata.get("case_name") or str(case_no)
            md = dict(d.metadata)
            md["__expanded"] = True
            expanded_cases.append(
                Document(
                    page_content=f"[판례 전문: {title}]\n{full_text}",
                    metadata=md,
                )
            )
        
        final_case = expanded_cases[:cfg.k_case]
        
        # 7) Priority sort
        final_docs = final_law + final_rule + final_case
        final_docs = sorted(
            final_docs,
            key=lambda x: _safe_int((x.metadata or {}).get("priority", 99), 99)
        )
        
        logger.info(
            f"📊 최종 검색 결과: Law={len(final_law)}, "
            f"Rule={len(final_rule)}, Case={len(final_case)}"
        )
        
        return final_docs
    
    # ----------------------------
    # Context Formatting
    # ----------------------------
    @staticmethod
    def format_context_with_hierarchy(docs: List[Document]) -> str:
        """검색된 문서를 법적 위계에 따라 섹션별로 재구성"""
        section_1_law: List[str] = []
        section_2_rule: List[str] = []
        section_3_case: List[str] = []
        
        for doc in docs:
            md = doc.metadata or {}
            p = _safe_int(md.get("priority", 99), 99)
            src = md.get("src_title", md.get("__source_index", "자료"))
            title = md.get("title", "")
            content = doc.page_content or ""
            
            entry = f"[{src}] {title}\n{content}".strip()
            
            if p in (1, 2, 4, 5):
                section_1_law.append(entry)
            elif p in (3, 6, 7, 8, 11):
                section_2_rule.append(entry)
            else:
                section_3_case.append(entry)
        
        parts: List[str] = []
        if section_1_law:
            parts.append(
                "## [SECTION 1: 핵심 법령 (최우선 법적 근거)]\n"
                + "\n\n".join(section_1_law)
            )
        if section_2_rule:
            parts.append(
                "## [SECTION 2: 관련 규정 및 절차 (세부 기준)]\n"
                + "\n\n".join(section_2_rule)
            )
        if section_3_case:
            parts.append(
                "## [SECTION 3: 판례 및 해석 사례 (적용 예시)]\n"
                + "\n\n".join(section_3_case)
            )
        
        return "\n\n".join(parts).strip()
    
    # ----------------------------
    # Answer Generation
    # ----------------------------
    def generate_answer(
        self,
        user_input: str,
        *,
        skip_normalization: bool = False
    ) -> str:
        """
        최종 답변 생성 (GPT-4o-mini 사용)
        
        Args:
            user_input: 사용자 질문
            skip_normalization: 질문 표준화 건너뛰기
        
        Returns:
            생성된 답변
        """
        # 1) Normalize (Solar-Pro2)
        normalized_query = (
            user_input if skip_normalization
            else self.normalize_query(user_input)
        )
        if not skip_normalization:
            logger.info(f"🔄 표준화된 질문: {normalized_query}")
        
        # 2) Retrieve (Hybrid)
        retrieved_docs = self.triple_hybrid_retrieval(normalized_query)
        if not retrieved_docs:
            return "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다."
        
        # 3) Context
        hierarchical_context = self.format_context_with_hierarchy(retrieved_docs)
        
        # 4) Generate (GPT-4o-mini)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | self._generation_llm | StrOutputParser()
        
        logger.info("🤖 답변 생성 중 (GPT-4o-mini)...")
        try:
            return str(chain.invoke({
                "context": hierarchical_context,
                "question": normalized_query
            })).strip()
        except Exception as e:
            logger.error(f"⚠️ 답변 생성 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."


# --------------------------------------------------------------------------------------
# Convenience Functions
# --------------------------------------------------------------------------------------
def create_pipeline(
    enable_hybrid: bool = True,
    hybrid_method: str = "rrf",
    enable_rerank: bool = True,
    **kwargs
) -> RAGPipeline:
    """
    파이프라인 생성 헬퍼 함수
    
    Examples:
        # 기본 설정
        pipeline = create_pipeline()
        
        # Hybrid 비활성화
        pipeline = create_pipeline(enable_hybrid=False)
        
        # Rerank 비활성화
        pipeline = create_pipeline(enable_rerank=False)
    """
    config = RAGConfig(
        enable_hybrid=enable_hybrid,
        hybrid_method=hybrid_method,
        enable_rerank=enable_rerank,
        **kwargs
    )
    return RAGPipeline(config)


# --------------------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------------------
__all__ = [
    # Config & Pipeline
    "RAGConfig",
    "RAGPipeline",
    "create_pipeline",
    # Tokenizers
    "Tokenizer",
    "SimpleTokenizer",
    "KiwiTokenizer",
    "get_default_tokenizer",
    # BM25
    "BM25Scorer",
    # Fusion
    "ScoreFusion",
    # Constants
    "INDEX_NAMES",
    "KEYWORD_DICT",
    "NORMALIZATION_PROMPT",
    "SYSTEM_PROMPT",
    # Availability Flags
    "RANK_BM25_AVAILABLE",
    "KIWI_AVAILABLE",
    "COHERE_AVAILABLE",
    "UPSTAGE_CHAT_AVAILABLE",
    "OPENAI_AVAILABLE",
    "OLLAMA_AVAILABLE",
]


# --------------------------------------------------------------------------------------
# Main (Test)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Unified RAG Pipeline 테스트")
    print("   - Normalize: Upstage Solar-Pro2")
    print("   - Generate: OpenAI GPT-4o-mini")
    print("=" * 70)
    
    # 의존성 체크
    print("\n📦 의존성 상태:")
    print(f"  - Upstage Chat: {'✅' if UPSTAGE_CHAT_AVAILABLE else '❌'}")
    print(f"  - OpenAI: {'✅' if OPENAI_AVAILABLE else '❌'}")
    print(f"  - Ollama (Fallback): {'✅' if OLLAMA_AVAILABLE else '❌'}")
    print(f"  - rank_bm25: {'✅' if RANK_BM25_AVAILABLE else '❌ (builtin 사용)'}")
    print(f"  - kiwipiepy: {'✅' if KIWI_AVAILABLE else '❌ (SimpleTokenizer 사용)'}")
    print(f"  - cohere: {'✅' if COHERE_AVAILABLE else '❌'}")
    
    try:
        print("\n🔧 파이프라인 초기화 중...")
        pipeline = create_pipeline()
        
        test_queries = [
            "집주인이 보증금을 안 돌려줘요. 어떻게 해야 하나요?",
            "전입신고는 언제까지 해야 대항력이 생기나요?",
            "계약 갱신을 요구했는데 집주인이 거절했어요.",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"📝 테스트 {i}: {query}")
            print("=" * 70)
            
            answer = pipeline.generate_answer(query)
            print(f"\n💬 답변:\n{answer}")
        
    except Exception as e:
        logger.error(f"🔥 에러 발생: {e}")
        raise
