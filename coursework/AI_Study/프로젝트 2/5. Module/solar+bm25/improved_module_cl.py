"""
Hybrid RAG Module - Dense (Solar) + Sparse (BM25) 통합 검색

주택임대차 RAG 시스템 - 하이브리드 검색 및 답변 생성 모듈

[핵심 개선사항]
- Dense 검색 (Solar/Cohere Embedding via Pinecone)
- Sparse 검색 (BM25 with Korean tokenization)
- Reciprocal Rank Fusion (RRF) 또는 Weighted Score 결합
- 2-stage case expansion 유지
- Cohere Rerank 선택적 적용

[하이브리드 검색 전략]
1. Dense: Pinecone VectorStore에서 시맨틱 유사도 기반 검색
2. Sparse: BM25로 키워드 매칭 기반 검색 (검색된 문서 풀에서 재순위화)
3. Fusion: RRF 또는 가중 평균으로 두 결과 결합
4. Rerank: (선택) Cohere로 최종 관련도 기반 재순위화

[의존성]
- rank_bm25: BM25 스코어링
- kiwipiepy (선택): 한국어 형태소 분석 (없으면 공백 토큰화)
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Sequence, Tuple, Iterable, 
    Callable, Protocol, Union, Any
)

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore

# BM25
try:
    from rank_bm25 import BM25Okapi, BM25Plus
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None
    BM25Plus = None

# Korean tokenizer (optional)
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    Kiwi = None

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
# Index names
# --------------------------------------------------------------------------------------
INDEX_NAMES: Dict[str, str] = {
    "law": "law-index-final",
    "rule": "rule-index-final",
    "case": "case-index-final",
}

# --------------------------------------------------------------------------------------
# Keyword dictionary (query normalization)
# --------------------------------------------------------------------------------------
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
변경된 질문:
"""

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

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _safe_int(x: object, default: int = 99) -> int:
    try:
        return int(x)  # type: ignore[arg-type]
    except Exception:
        return default


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _dedupe_docs(
    docs: Iterable[Document],
    key_fields: Sequence[str] = ("chunk_id", "id"),
) -> List[Document]:
    """메타데이터 기반으로 중복 제거"""
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
# Korean Tokenizer
# --------------------------------------------------------------------------------------
class Tokenizer(ABC):
    """토크나이저 추상 클래스"""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


class SimpleTokenizer(Tokenizer):
    """공백 기반 단순 토크나이저 (fallback)"""
    
    def __init__(self, min_length: int = 1):
        self.min_length = min_length
        # 한글, 영문, 숫자만 추출
        self._pattern = re.compile(r'[가-힣a-zA-Z0-9]+')
    
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = self._pattern.findall(text.lower())
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
        # 기본: 명사, 동사, 형용사, 외래어/한자
        self.pos_tags = pos_tags or ('NNG', 'NNP', 'VV', 'VA', 'SL', 'SH')
        self.min_length = min_length
    
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        
        tokens = []
        result = self._kiwi.tokenize(text)
        for token in result:
            if token.tag in self.pos_tags and len(token.form) >= self.min_length:
                tokens.append(token.form.lower())
        return tokens


def get_default_tokenizer() -> Tokenizer:
    """사용 가능한 최적의 토크나이저 반환"""
    if KIWI_AVAILABLE:
        logger.info("✅ Kiwi 토크나이저 사용 (한국어 형태소 분석)")
        return KiwiTokenizer()
    else:
        logger.info("ℹ️ SimpleTokenizer 사용 (공백 기반)")
        return SimpleTokenizer()


# --------------------------------------------------------------------------------------
# BM25 Scorer
# --------------------------------------------------------------------------------------
class BM25Scorer:
    """
    BM25 기반 문서 스코어링
    
    검색된 문서 풀에서 쿼리와의 BM25 유사도를 계산합니다.
    """
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        algorithm: str = "okapi",  # "okapi" or "plus"
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not BM25_AVAILABLE:
            raise ImportError("rank_bm25가 설치되지 않았습니다: pip install rank-bm25")
        
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.algorithm = algorithm
        self.k1 = k1
        self.b = b
        
        self._bm25: Optional[Any] = None
        self._corpus_tokens: List[List[str]] = []
    
    def fit(self, documents: List[Document]) -> "BM25Scorer":
        """문서 코퍼스로 BM25 인덱스 구축"""
        self._corpus_tokens = [
            self.tokenizer.tokenize(doc.page_content or "")
            for doc in documents
        ]
        
        BM25Class = BM25Plus if self.algorithm == "plus" else BM25Okapi
        self._bm25 = BM25Class(self._corpus_tokens, k1=self.k1, b=self.b)
        
        return self
    
    def score(self, query: str) -> List[float]:
        """쿼리에 대한 각 문서의 BM25 점수 반환"""
        if self._bm25 is None:
            raise RuntimeError("fit()을 먼저 호출하세요")
        
        query_tokens = self.tokenizer.tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        return scores.tolist()
    
    def get_top_k(
        self, 
        query: str, 
        documents: List[Document], 
        k: int
    ) -> List[Tuple[Document, float]]:
        """상위 k개 문서와 점수 반환"""
        self.fit(documents)
        scores = self.score(query)
        
        # (document, score) 쌍으로 정렬
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:k]


# --------------------------------------------------------------------------------------
# Hybrid Score Fusion
# --------------------------------------------------------------------------------------
class ScoreFusion:
    """Dense와 Sparse 점수를 결합하는 전략"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        dense_ranks: Dict[str, int],  # doc_id -> rank (1-indexed)
        sparse_ranks: Dict[str, int],
        k: int = 60,
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion (RRF)
        
        RRF Score = Σ 1/(k + rank)
        
        Args:
            dense_ranks: Dense 검색 결과의 순위
            sparse_ranks: Sparse 검색 결과의 순위
            k: RRF 상수 (기본값 60)
        
        Returns:
            doc_id -> RRF score
        """
        all_docs = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        scores: Dict[str, float] = {}
        
        for doc_id in all_docs:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (k + sparse_ranks[doc_id])
            scores[doc_id] = score
        
        return scores
    
    @staticmethod
    def weighted_sum(
        dense_scores: Dict[str, float],
        sparse_scores: Dict[str, float],
        alpha: float = 0.5,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        가중 합산
        
        Final Score = alpha * dense_score + (1-alpha) * sparse_score
        
        Args:
            dense_scores: Dense 검색 점수 (정규화 권장)
            sparse_scores: Sparse 검색 점수 (정규화 권장)
            alpha: Dense 가중치 (0~1)
            normalize: 점수 정규화 여부
        """
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
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """Min-Max 정규화"""
        if not scores:
            return scores
        
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class RAGConfig:
    """RAG 파이프라인 설정"""
    
    # LLM
    llm_model: str = "exaone3.5:2.4b"
    temperature: float = 0.1
    normalize_temperature: float = 0.0

    # Embedding
    embedding_model: str = "solar-embedding-1-large-passage"

    # Retrieval sizes (final target)
    k_law: int = 5
    k_rule: int = 5
    k_case: int = 3

    # Oversampling before fusion/rerank
    search_multiplier: int = 2

    # ============ Hybrid Search Settings ============
    enable_hybrid: bool = True
    hybrid_method: str = "rrf"  # "rrf" or "weighted"
    hybrid_alpha: float = 0.5   # Dense 가중치 (weighted 방식에서 사용)
    rrf_k: int = 60             # RRF 상수
    
    # BM25 Settings
    bm25_algorithm: str = "okapi"  # "okapi" or "plus"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    use_kiwi_tokenizer: bool = True  # False면 SimpleTokenizer 사용
    
    # ============ Rerank Settings ============
    enable_rerank: bool = True
    rerank_threshold: float = 0.2
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_max_documents: int = 80
    rerank_doc_max_chars: int = 2000

    # 2-stage case expansion
    case_candidate_k: int = 40
    case_expand_top_n: Optional[int] = None
    case_context_top_k: int = 50

    # Deduping
    dedupe_key_fields: Tuple[str, ...] = ("chunk_id", "id")

    def __post_init__(self) -> None:
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature는 0~2 사이여야 합니다.")
        if not (0 <= self.rerank_threshold <= 1):
            raise ValueError("rerank_threshold는 0~1 사이여야 합니다.")
        if not (0 <= self.hybrid_alpha <= 1):
            raise ValueError("hybrid_alpha는 0~1 사이여야 합니다.")
        if self.hybrid_method not in ("rrf", "weighted"):
            raise ValueError("hybrid_method는 'rrf' 또는 'weighted'여야 합니다.")


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
class RAGPipeline:
    """
    Hybrid RAG Pipeline - Dense (Solar) + Sparse (BM25)
    
    Usage:
        # 기본 (하이브리드 활성화)
        pipeline = RAGPipeline()
        answer = pipeline.generate_answer("보증금을 못 돌려받았어요")
        
        # Dense만 사용
        config = RAGConfig(enable_hybrid=False)
        pipeline = RAGPipeline(config)
        
        # 가중 합산 방식
        config = RAGConfig(hybrid_method="weighted", hybrid_alpha=0.7)
        pipeline = RAGPipeline(config)
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        *,
        pc_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        embedding: Optional[object] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.config = config or RAGConfig()

        self._pc_api_key = pc_api_key or os.getenv("PINECONE_API_KEY")
        self._cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")

        if not self._pc_api_key:
            raise ValueError(
                "Pinecone API key가 필요합니다. "
                "pc_api_key 인자 또는 PINECONE_API_KEY 환경변수를 설정하세요."
            )

        # Embeddings
        if embedding is not None:
            self._embedding = embedding
        else:
            self._embedding = UpstageEmbeddings(
                model=self.config.embedding_model
            )

        # Vector stores
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

        # LLM instances
        self._normalize_llm = ChatOllama(
            model=self.config.llm_model,
            temperature=self.config.normalize_temperature,
        )
        self._generation_llm = ChatOllama(
            model=self.config.llm_model,
            temperature=self.config.temperature,
        )

        # Tokenizer for BM25
        self._tokenizer: Optional[Tokenizer] = None
        if self.config.enable_hybrid:
            if tokenizer is not None:
                self._tokenizer = tokenizer
            elif self.config.use_kiwi_tokenizer and KIWI_AVAILABLE:
                self._tokenizer = KiwiTokenizer()
                logger.info("✅ Kiwi 토크나이저 초기화 완료")
            else:
                self._tokenizer = SimpleTokenizer()
                logger.info("ℹ️ SimpleTokenizer 사용 (공백 기반)")
            
            if not BM25_AVAILABLE:
                logger.warning(
                    "⚠️ rank_bm25가 설치되지 않아 하이브리드 검색이 비활성화됩니다. "
                    "설치: pip install rank-bm25"
                )
                self.config.enable_hybrid = False

        # Cohere client for rerank
        self._cohere_client: Optional[Any] = None
        if self.config.enable_rerank:
            if not COHERE_AVAILABLE:
                logger.warning("⚠️ cohere 패키지가 없어 rerank를 비활성화합니다.")
            elif not self._cohere_api_key:
                logger.warning("⚠️ COHERE_API_KEY가 없어 rerank를 비활성화합니다.")
            else:
                self._cohere_client = cohere.Client(self._cohere_api_key)
                logger.info("✅ Cohere Reranking 활성화")

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
    # Core methods
    # ----------------------------
    def normalize_query(self, user_query: str) -> str:
        """사용자 질문을 법률 용어로 표준화"""
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

    def _compute_bm25_scores(
        self, 
        query: str, 
        documents: List[Document]
    ) -> Dict[str, float]:
        """BM25 점수 계산"""
        if not documents or not self._tokenizer:
            return {}
        
        scorer = BM25Scorer(
            tokenizer=self._tokenizer,
            algorithm=self.config.bm25_algorithm,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        scorer.fit(documents)
        scores = scorer.score(query)
        
        return {
            self._get_doc_id(doc): score
            for doc, score in zip(documents, scores)
        }

    def _hybrid_fusion(
        self,
        query: str,
        dense_docs: List[Document],
    ) -> List[Document]:
        """
        Dense 검색 결과에 BM25를 결합하여 하이브리드 순위 생성
        
        Args:
            query: 검색 쿼리
            dense_docs: Dense 검색으로 가져온 문서들
        
        Returns:
            하이브리드 점수로 재순위화된 문서 리스트
        """
        if not self.config.enable_hybrid or not dense_docs:
            return dense_docs
        
        cfg = self.config
        
        # Dense ranks (순위 기반)
        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        for rank, doc in enumerate(dense_docs, start=1):
            doc_id = self._get_doc_id(doc)
            dense_ranks[doc_id] = rank
            # Dense score는 순위의 역수로 근사
            dense_scores[doc_id] = 1.0 / rank
        
        # BM25 scores
        bm25_scores = self._compute_bm25_scores(query, dense_docs)
        
        # BM25 ranks
        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(sorted_bm25, start=1)}
        
        # Fusion
        if cfg.hybrid_method == "rrf":
            fused_scores = ScoreFusion.reciprocal_rank_fusion(
                dense_ranks, bm25_ranks, k=cfg.rrf_k
            )
        else:  # weighted
            fused_scores = ScoreFusion.weighted_sum(
                dense_scores, bm25_scores, alpha=cfg.hybrid_alpha
            )
        
        # 문서를 fused_score로 재정렬
        doc_map = {self._get_doc_id(d): d for d in dense_docs}
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        reordered = [doc_map[doc_id] for doc_id in sorted_ids if doc_id in doc_map]
        
        logger.info(
            f"🔀 Hybrid Fusion 완료 ({cfg.hybrid_method}): "
            f"{len(dense_docs)}개 → {len(reordered)}개"
        )
        
        return reordered

    def _rerank(
        self, 
        query: str, 
        docs: List[Document]
    ) -> Optional[List[Tuple[int, float]]]:
        """Cohere rerank 실행"""
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
            ranked = [
                (r.index, float(r.relevance_score)) 
                for r in rerank_results.results
            ]
            return ranked
        except Exception as e:
            logger.warning(f"⚠️ Rerank 실패 (skip): {e}")
            return None

    def _cap_for_rerank(
        self, 
        law: List[Document], 
        rule: List[Document], 
        case: List[Document]
    ) -> List[Document]:
        """rerank 입력 문서 수 제한"""
        cfg = self.config
        law = _dedupe_docs(law, cfg.dedupe_key_fields)
        rule = _dedupe_docs(rule, cfg.dedupe_key_fields)
        case = _dedupe_docs(case, cfg.dedupe_key_fields)

        base = law + rule
        if len(base) >= cfg.rerank_max_documents:
            return base[: cfg.rerank_max_documents]

        remaining = cfg.rerank_max_documents - len(base)
        return base + case[:remaining]

    def triple_hybrid_retrieval(self, query: str) -> List[Document]:
        """
        3중 인덱스 하이브리드 검색 (Dense + Sparse + Rerank)
        
        검색 흐름:
        1. Dense 검색 (Pinecone - Solar Embedding)
        2. Hybrid Fusion (Dense + BM25)
        3. Rerank (Cohere, 선택적)
        4. 2-stage Case Expansion
        5. Priority 정렬
        """
        cfg = self.config
        mult = cfg.search_multiplier

        logger.info(f"🔍 [Hybrid 검색] query='{query}'")

        # 1) Dense Retrieval (oversampling)
        docs_law = self._attach_source(
            self.law_store.similarity_search(query, k=cfg.k_law * mult),
            "law",
        )
        docs_rule = self._attach_source(
            self.rule_store.similarity_search(query, k=cfg.k_rule * mult),
            "rule",
        )
        docs_case_chunks = self._attach_source(
            self.case_store.similarity_search(query, k=cfg.case_candidate_k),
            "case",
        )

        # 2) Hybrid Fusion (Dense + BM25) - 각 인덱스별로 적용
        if cfg.enable_hybrid:
            docs_law = self._hybrid_fusion(query, docs_law)
            docs_rule = self._hybrid_fusion(query, docs_rule)
            docs_case_chunks = self._hybrid_fusion(query, docs_case_chunks)

        # 3) Prepare for rerank
        combined_for_rerank = self._cap_for_rerank(docs_law, docs_rule, docs_case_chunks)

        # 4) Rerank (optional)
        selected_docs: List[Document]
        ranked = self._rerank(query, combined_for_rerank) if cfg.enable_rerank else None

        if ranked:
            filtered = [(i, s) for (i, s) in ranked if s >= cfg.rerank_threshold]
            if not filtered:
                desired = min(cfg.k_law + cfg.k_rule + cfg.k_case, len(ranked))
                filtered = ranked[:desired]
            selected_docs = [combined_for_rerank[i] for (i, _s) in filtered]
            logger.info(f"📌 Rerank 완료: {len(selected_docs)}개 선택 (threshold={cfg.rerank_threshold})")
        else:
            selected_docs = combined_for_rerank

        # 5) Deduplicate
        selected_docs = _dedupe_docs(selected_docs, cfg.dedupe_key_fields)

        # 6) Select top docs per source + 2-stage case expansion
        law_ranked = [d for d in selected_docs if d.metadata.get("__source_index") == "law"]
        rule_ranked = [d for d in selected_docs if d.metadata.get("__source_index") == "rule"]
        case_ranked_chunks = [d for d in selected_docs if d.metadata.get("__source_index") == "case"]

        final_law = law_ranked[: cfg.k_law]
        final_rule = rule_ranked[: cfg.k_rule]

        # Case expansion
        top_n = cfg.case_expand_top_n if cfg.case_expand_top_n is not None else cfg.k_case
        seen_case_no: set = set()
        chosen_case_docs: List[Document] = []
        
        for d in case_ranked_chunks:
            case_no = d.metadata.get("case_no")
            if not case_no or case_no in seen_case_no:
                continue
            seen_case_no.add(case_no)
            chosen_case_docs.append(d)
            if len(chosen_case_docs) >= top_n:
                break

        expanded_cases: List[Document] = []
        for d in chosen_case_docs:
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

        final_case = expanded_cases[: cfg.k_case]

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
    # Context formatting
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
    # Answer generation
    # ----------------------------
    def generate_answer(
        self, 
        user_input: str, 
        *, 
        skip_normalization: bool = False
    ) -> str:
        """
        최종 답변 생성
        
        Args:
            user_input: 사용자 질문
            skip_normalization: 질문 표준화 건너뛰기
        
        Returns:
            생성된 답변
        """
        # 1) Normalize
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

        # 4) Generate
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | self._generation_llm | StrOutputParser()

        logger.info("🤖 답변 생성 중...")
        try:
            return str(chain.invoke({
                "context": hierarchical_context, 
                "question": normalized_query
            })).strip()
        except Exception as e:
            logger.warning(f"⚠️ 답변 생성 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."


# --------------------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------------------
def create_pipeline(
    enable_hybrid: bool = True,
    hybrid_method: str = "rrf",
    hybrid_alpha: float = 0.5,
    enable_rerank: bool = True,
    **kwargs
) -> RAGPipeline:
    """
    파이프라인 생성 헬퍼 함수
    
    Examples:
        # 기본 하이브리드 (RRF)
        pipeline = create_pipeline()
        
        # Dense만 사용
        pipeline = create_pipeline(enable_hybrid=False)
        
        # 가중 합산 (Dense 70%)
        pipeline = create_pipeline(hybrid_method="weighted", hybrid_alpha=0.7)
        
        # Rerank 비활성화
        pipeline = create_pipeline(enable_rerank=False)
    """
    config = RAGConfig(
        enable_hybrid=enable_hybrid,
        hybrid_method=hybrid_method,
        hybrid_alpha=hybrid_alpha,
        enable_rerank=enable_rerank,
        **kwargs
    )
    return RAGPipeline(config)


# --------------------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------------------
__all__ = [
    # Config
    "RAGConfig",
    # Pipeline
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
    # Availability flags
    "BM25_AVAILABLE",
    "KIWI_AVAILABLE",
    "COHERE_AVAILABLE",
]


# --------------------------------------------------------------------------------------
# Main (Test)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Hybrid RAG Pipeline (Dense + Sparse) 테스트")
    print("=" * 70)
    
    # 의존성 체크
    print("\n📦 의존성 상태:")
    print(f"  - rank_bm25: {'✅ 사용 가능' if BM25_AVAILABLE else '❌ 미설치'}")
    print(f"  - kiwipiepy: {'✅ 사용 가능' if KIWI_AVAILABLE else '❌ 미설치 (SimpleTokenizer 사용)'}")
    print(f"  - cohere: {'✅ 사용 가능' if COHERE_AVAILABLE else '❌ 미설치'}")
    
    try:
        # 파이프라인 생성
        print("\n🔧 파이프라인 초기화 중...")
        pipeline = create_pipeline(
            enable_hybrid=True,
            hybrid_method="rrf",
            enable_rerank=True,
        )
        
        # 테스트 쿼리
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