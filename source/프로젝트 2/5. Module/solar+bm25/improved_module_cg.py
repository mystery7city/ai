"""
Unified RAG module (No FastAPI integration)

주택임대차 RAG 시스템 - 통합 검색 및 답변 생성 모듈

핵심 설계
- RAGConfig: 설정을 중앙 관리
- RAGPipeline: (1) 질문 표준화 → (2) 3중 인덱스 검색 → (3) 선택적 Rerank → (4) 2-stage 판례 확장 → (5) 법적 위계 컨텍스트 구성 → (6) 답변 생성
- 2-stage case 전략:
  1) case-index는 '청크' 단위로 먼저 후보를 많이 가져와 rerank/선별
  2) 최종 선택된 상위 사건번호(top N)에 대해서만 전문(context) 확장

필수 외부 의존성(기본 경로)
- langchain_core, langchain_community, langchain_pinecone
- cohere (선택: rerank 사용 시 필요)
- Pinecone 인덱스 3개: law/rule/case (INDEX_NAMES 참고)

환경변수
- PINECONE_API_KEY: PineconeVectorStore 접근용
- UPSTAGE_API_KEY: UpstageEmbeddings(SOLAR embedding)용 (upstage 임베딩 사용 시)
- COHERE_API_KEY: CohereEmbeddings / Cohere Rerank용 (cohere 임베딩 또는 rerank 사용 시)

작성: unified from rag_module_cl2.py (+ 2-stage case rerank 개선, 레거시/프레임워크 가이드 제거)
"""

from __future__ import annotations

import logging
import os
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Iterable, Callable, Any, Mapping

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Embeddings backends (SOLAR: Upstage, fallback: Cohere)
try:
    from langchain_upstage import UpstageEmbeddings  # type: ignore
    UPSTAGE_AVAILABLE = True
except Exception:
    UpstageEmbeddings = None  # type: ignore
    UPSTAGE_AVAILABLE = False

try:
    from langchain_community.embeddings import CohereEmbeddings  # type: ignore
    COHERE_EMBED_AVAILABLE = True
except Exception:
    CohereEmbeddings = None  # type: ignore
    COHERE_EMBED_AVAILABLE = False

from langchain_pinecone import PineconeVectorStore

# Optional: Cohere Rerank
try:
    import cohere  # type: ignore
    COHERE_AVAILABLE = True
except Exception:
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
    "law": "law-index-final",    # Priority 1,2,4,5: 주임법, 민법 등 핵심 법률
    "rule": "rule-index-final",  # Priority 3,6,7,8,11: 시행규칙, 조례, 절차
    "case": "case-index-final",  # Priority 9: 판례, 상담사례
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

    # 3. 기간 및 종료/갱신
    "재계약": "계약갱신", "연장": "계약갱신", "갱신": "계약갱신",
    "갱신청구": "계약갱신요구권", "2년더": "계약갱신요구권",
    "자동연장": "묵시적갱신", "묵시": "묵시적갱신",
    "이사": "주택의인도", "짐빼기": "주택의인도", "퇴거": "주택의인도",
    "방빼": "계약해지",
    "주소옮기기": "주민등록", "전입신고": "주민등록", "주소지이전": "주민등록",
    "집주인바뀜": "임대인지위승계", "주인바뀜": "임대인지위승계",
    "매매": "임대인지위승계",
    "나가라고함": "계약갱신거절", "쫓겨남": "명도", "비워달라": "명도",
    "중도해지": "계약해지",

    # 4. 수리 및 생활환경
    "집고치기": "수선의무", "수리": "수선의무", "고쳐줘": "수선의무",
    "안고쳐줌": "수선의무위반",
    "곰팡이": "하자", "물샘": "누수", "보일러고장": "하자", "파손": "훼손",
    "깨끗이치우기": "원상회복의무", "원래대로해놓기": "원상회복",
    "청소비": "원상회복비용", "청소": "원상회복",
    "층간소음": "공동생활평온", "옆집소음": "방음", "개키우기": "반려동물특약",

    # 5. 권리/대항력/확정일자
    "확정일자": "확정일자", "전입": "주민등록", "대항력": "대항력",
    "우선변제": "우선변제권", "최우선": "최우선변제권",
    "경매": "경매절차", "공매": "공매절차",
    "등기": "등기부등본", "등본": "등기부등본",
    "근저당": "근저당권", "가압류": "가압류", "가처분": "가처분",

    # 6. 분쟁 해결
    "내용증명": "내용증명", "소송": "소송", "민사": "민사소송",
    "조정위": "주택임대차분쟁조정위원회", "소송말고": "분쟁조정",
    "법원가기싫음": "분쟁조정",
    "집주인사망": "임차권승계", "자식상속": "임차권승계",
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
    """
    메타데이터 기반으로 중복 제거 (chunk_id 우선)
    - key_fields 중 첫 번째로 발견되는 값을 키로 사용
    - 키가 없으면 page_content의 해시(짧게)로 폴백
    """
    seen = set()
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
# Sparse scoring (BM25) + fusion utilities
# --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _default_tokenize(text: str) -> List[str]:
    # 한국어/영문/숫자 중심의 가벼운 토크나이저 (형태소 분석기 없이도 동작)
    return _TOKEN_RE.findall((text or "").lower())


def _bm25_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    '''
    BM25Okapi-lite (candidate-level).
    - docs_tokens는 후보 문서들(보통 20~80개)에 대해서만 계산하므로 O(N*V)로도 충분.
    '''
    N = len(docs_tokens)
    if N == 0:
        return []
    if not query_tokens:
        return [0.0] * N

    # document lengths
    doc_lens = [len(toks) for toks in docs_tokens]
    avgdl = (sum(doc_lens) / N) if N else 1.0
    if avgdl <= 0:
        avgdl = 1.0

    # df: term -> number of docs containing term
    df: Dict[str, int] = defaultdict(int)
    for toks in docs_tokens:
        seen = set(toks)
        for t in seen:
            df[t] += 1

    # idf
    idf: Dict[str, float] = {}
    for t, dfi in df.items():
        # standard BM25 idf variant
        idf[t] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

    # query term frequency (optional weighting)
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


def _rank_fusion(
    dense_ranks: List[int],
    sparse_ranks: List[int],
    *,
    mode: str = "rrf",
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    rrf_k: int = 60,
) -> List[float]:
    '''
    Rank 기반 fusion (점수 스케일/거리/유사도 정의에 덜 민감).
    Returns:
        fused_scores (higher is better)
    '''
    n = len(dense_ranks)
    if n == 0:
        return []
    if mode == "rrf":
        k = max(1, int(rrf_k))
        return [
            (w_dense / (k + dense_ranks[i])) + (w_sparse / (k + sparse_ranks[i]))
            for i in range(n)
        ]
    # mode == "rank_sum": ranks -> [0,1]로 변환 후 가중합
    if n == 1:
        return [w_dense + w_sparse]
    def to_unit(r: int) -> float:
        return 1.0 - (r - 1) / (n - 1)
    return [
        (w_dense * to_unit(dense_ranks[i])) + (w_sparse * to_unit(sparse_ranks[i]))
        for i in range(n)
    ]



# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class RAGConfig:
    # LLM
    llm_model: str = "exaone3.5:2.4b"
    temperature: float = 0.1
    normalize_temperature: float = 0.0

    # Embedding
    # - "auto": UPSTAGE_AVAILABLE면 Upstage(SOLAR) 우선, 아니면 Cohere
    # - "upstage": UpstageEmbeddings 강제
    # - "cohere": CohereEmbeddings 강제
    embedding_backend: str = "auto"
    embedding_model: str = "solar-embedding-1-large-passage"

    # Retrieval sizes (final target)
    k_law: int = 5
    k_rule: int = 5
    k_case: int = 3

    # Oversampling before rerank
    search_multiplier: int = 2

    # Rerank
    enable_rerank: bool = True
    rerank_threshold: float = 0.2
    rerank_model: str = "rerank-multilingual-v3.0"
    rerank_max_documents: int = 80              # cohere rerank 입력 문서 최대 개수
    rerank_doc_max_chars: int = 2000            # rerank 입력 문서 truncation


    # Dense + Sparse hybrid (BM25)
    # - Pinecone 인덱스 변경 없이도, Dense 상위 후보에 대해 BM25 점수를 계산하여 결합(=candidate-level hybrid)합니다.
    enable_bm25: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_max_doc_chars: int = 4000  # BM25 토크나이징/스코어링 시 문서 텍스트 최대 길이

    # Fusion strategy: "rrf" (권장, 점수 스케일에 강건) | "rank_sum"
    hybrid_fusion: str = "rrf"
    hybrid_dense_weight: float = 0.6
    hybrid_sparse_weight: float = 0.4
    rrf_k: int = 60

    # 2-stage case expansion
    case_candidate_k: int = 40                  # case-index에서 '청크'로 가져올 후보 수
    case_expand_top_n: Optional[int] = None     # None이면 k_case 사용
    case_context_top_k: int = 50                # 선택된 사건번호의 전문 확장 시 최대 chunk 수

    # Deduping
    dedupe_key_fields: Tuple[str, ...] = ("chunk_id", "id")

    def __post_init__(self) -> None:
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature는 0~2 사이여야 합니다.")
        if not (0 <= self.rerank_threshold <= 1):
            raise ValueError("rerank_threshold는 0~1 사이여야 합니다.")
        if self.search_multiplier < 1:
            raise ValueError("search_multiplier는 1 이상이어야 합니다.")
        if self.rerank_max_documents < 1:
            raise ValueError("rerank_max_documents는 1 이상이어야 합니다.")
        if self.case_candidate_k < 1:
            raise ValueError("case_candidate_k는 1 이상이어야 합니다.")
        if self.case_context_top_k < 1:
            raise ValueError("case_context_top_k는 1 이상이어야 합니다.")

        # BM25 / hybrid fusion validation
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1은 0보다 커야 합니다.")
        if not (0 <= self.bm25_b <= 1):
            raise ValueError("bm25_b는 0~1 사이여야 합니다.")
        if self.bm25_max_doc_chars < 200:
            raise ValueError("bm25_max_doc_chars는 200 이상을 권장합니다.")
        if self.hybrid_fusion not in ("rrf", "rank_sum"):
            raise ValueError('hybrid_fusion은 "rrf" 또는 "rank_sum" 이어야 합니다.')
        if self.rrf_k < 1:
            raise ValueError("rrf_k는 1 이상이어야 합니다.")
        if self.hybrid_dense_weight < 0 or self.hybrid_sparse_weight < 0:
            raise ValueError("hybrid_*_weight는 0 이상이어야 합니다.")
        if self.hybrid_dense_weight == 0 and self.hybrid_sparse_weight == 0:
            raise ValueError("hybrid_dense_weight와 hybrid_sparse_weight가 모두 0일 수는 없습니다.")



# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
class RAGPipeline:
    """
    Unified RAG pipeline (no web framework integration).

    Usage:
        pipeline = RAGPipeline()
        answer = pipeline.generate_answer("보증금을 못 돌려받았어요. 어떻게 해야 하나요?")
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        *,
        pc_api_key: Optional[str] = None,
        upstage_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        embedding: Optional[object] = None,
        cohere_client: Optional[object] = None,
    ) -> None:
        self.config = config or RAGConfig()

        self._pc_api_key = pc_api_key or os.getenv("PINECONE_API_KEY")
        self._upstage_api_key = upstage_api_key or os.getenv("UPSTAGE_API_KEY")
        self._cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY")

        if not self._pc_api_key:
            raise ValueError("Pinecone API key가 필요합니다. pc_api_key 인자로 주거나 PINECONE_API_KEY를 설정하세요.")        # Embeddings
        # - 기본은 config.embedding_backend에 따라 자동 선택합니다.
        # - Solar(Upstage) + BM25 Dense/Sparse 조합을 원하면 embedding_backend="upstage" 권장
        if embedding is not None:
            self._embedding = embedding
        else:
            backend = (self.config.embedding_backend or "auto").lower()

            if backend in ("auto", "upstage") and UPSTAGE_AVAILABLE:
                if not self._upstage_api_key:
                    raise ValueError(
                        "embedding_backend=upstage(또는 auto에서 upstage 사용)를 위해서는 UPSTAGE_API_KEY가 필요합니다. "
                        "upstage_api_key 인자로 주거나 UPSTAGE_API_KEY를 설정하거나, embedding 객체를 주입하세요."
                    )
                # langchain_upstage는 보통 환경변수에서 API 키를 읽습니다.
                os.environ.setdefault("UPSTAGE_API_KEY", self._upstage_api_key)
                self._embedding = UpstageEmbeddings(model=self.config.embedding_model)  # type: ignore[call-arg]

            elif backend in ("auto", "cohere") and COHERE_EMBED_AVAILABLE:
                if not self._cohere_api_key:
                    raise ValueError(
                        "embedding_backend=cohere(또는 auto에서 cohere 사용)를 위해서는 COHERE_API_KEY가 필요합니다. "
                        "cohere_api_key 인자로 주거나 COHERE_API_KEY를 설정하거나, embedding 객체를 주입하세요."
                    )
                self._embedding = CohereEmbeddings(  # type: ignore[call-arg]
                    model=self.config.embedding_model,
                    cohere_api_key=self._cohere_api_key,
                )

            else:
                raise ImportError(
                    "사용 가능한 embedding backend가 없습니다. "
                    "langchain_upstage(UpstageEmbeddings) 또는 langchain_community(CohereEmbeddings) 설치/설정을 확인하세요."
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

        # LLM instances (reused)
        self._normalize_llm = ChatOllama(
            model=self.config.llm_model,
            temperature=self.config.normalize_temperature,
        )
        self._generation_llm = ChatOllama(
            model=self.config.llm_model,
            temperature=self.config.temperature,
        )

        # Cohere rerank client (optional)
        self._cohere_client = None
        if self.config.enable_rerank:
            if not COHERE_AVAILABLE:
                logger.warning("⚠️ cohere 패키지가 없어 rerank를 비활성화합니다.")
            elif not self._cohere_api_key:
                logger.warning("⚠️ COHERE_API_KEY가 없어 rerank를 비활성화합니다.")
            else:
                self._cohere_client = cohere_client or cohere.Client(self._cohere_api_key)  # type: ignore[attr-defined]

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
    # Core steps
    # ----------------------------
    def normalize_query(self, user_query: str) -> str:
        """사용자 질문을 법률 용어로 표준화"""
        prompt = ChatPromptTemplate.from_template(NORMALIZATION_PROMPT)
        chain = prompt | self._normalize_llm | StrOutputParser()

        try:
            normalized = chain.invoke({"dictionary": KEYWORD_DICT, "question": user_query})
            return str(normalized).strip()
        except Exception as e:
            logger.warning(f"⚠️ 전처리 실패 (원본 사용): {e}")
            return user_query

    def get_full_case_context(self, case_no: str) -> str:
        """특정 사건번호(case_no)의 판례 전문(청크들을 연결)을 가져옴"""
        try:
            results = self.case_store.similarity_search(
                query="판례 전문 검색",  # API 요구사항용 더미 쿼리
                k=self.config.case_context_top_k,
                filter={"case_no": {"$eq": case_no}},
            )
            # chunk_id 순 정렬 후 중복 제거
            sorted_docs = sorted(results, key=lambda x: str(x.metadata.get("chunk_id", "")))
            unique_docs = _dedupe_docs(sorted_docs, self.config.dedupe_key_fields)
            return "\n".join([d.page_content for d in unique_docs]).strip()
        except Exception as e:
            logger.warning(f"⚠️ 판례 전문 로딩 실패 ({case_no}): {e}")
            return ""

    def _attach_source(self, docs: List[Document], source: str) -> List[Document]:
        """검색 출처(law/rule/case)를 메타데이터에 주입"""
        for d in docs:
            if d.metadata is None:
                d.metadata = {}
            d.metadata["__source_index"] = source
        return docs

    def _rerank(self, query: str, docs: List[Document]) -> Optional[List[Tuple[int, float]]]:
        """
        Cohere rerank 실행
        Returns:
            [(doc_index, score), ...] (score desc)
            실패/비활성 시 None
        """
        if not self._cohere_client:
            return None

        cfg = self.config
        # cohere 문서 입력 준비 (너무 길면 truncation)
        texts = [_truncate(d.page_content or "", cfg.rerank_doc_max_chars) for d in docs]

        try:
            rerank_results = self._cohere_client.rerank(
                model=cfg.rerank_model,
                query=query,
                documents=texts,
                top_n=len(texts),
            )
            # 결과는 relevance_score 내림차순으로 제공되는 것이 일반적
            ranked = [(r.index, float(r.relevance_score)) for r in rerank_results.results]
            return ranked
        except Exception as e:
            logger.warning(f"⚠️ Rerank 실패 (skip): {e}")
            return None

    def _cap_for_rerank(self, law: List[Document], rule: List[Document], case: List[Document]) -> List[Document]:
        """
        rerank 입력 문서 수를 제한.
        - law/rule은 가능한 한 유지
        - case는 남는 슬롯만큼 채움
        """
        cfg = self.config
        law = _dedupe_docs(law, cfg.dedupe_key_fields)
        rule = _dedupe_docs(rule, cfg.dedupe_key_fields)
        case = _dedupe_docs(case, cfg.dedupe_key_fields)

        base = law + rule
        if len(base) >= cfg.rerank_max_documents:
            return base[: cfg.rerank_max_documents]

        remaining = cfg.rerank_max_documents - len(base)
        return base + case[:remaining]


    # ----------------------------
    # Dense + Sparse (BM25) candidate-level hybrid
    # ----------------------------
    def _search_dense_candidates(self, store: PineconeVectorStore, query: str, k: int) -> List[Document]:
        '''
        PineconeVectorStore에서 dense 검색을 수행하고, 가능한 경우 score를 메타데이터에 남깁니다.
        - score 스케일/의미(거리/유사도)는 구현/인덱스 설정에 따라 달라질 수 있으므로,
          후속 결합은 '랭크 기반'(RRF/RankSum)으로 처리합니다.
        '''
        try:
            pairs = store.similarity_search_with_score(query, k=k)  # type: ignore[attr-defined]
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

    def _dense_sparse_fuse(self, query: str, docs: List[Document]) -> List[Document]:
        '''
        Dense 결과(랭크) + BM25(sparse) 랭크를 결합하여 후보 리스트를 재정렬합니다.
        - 후보 수가 많지 않은 상황(보통 10~80개)에서 빠르게 동작합니다.
        - Pinecone 인덱스에 sparse vector를 별도로 저장하지 않아도 적용 가능합니다.
        '''
        cfg = self.config
        if not cfg.enable_bm25:
            return docs
        docs = _dedupe_docs(docs, cfg.dedupe_key_fields)
        if len(docs) <= 1:
            return docs

        # dense ranks: 메타데이터에 없으면 현재 순서를 사용
        dense_ranks: List[int] = []
        for i, d in enumerate(docs, start=1):
            if d.metadata is None:
                d.metadata = {}
            dense_ranks.append(int(d.metadata.get("__dense_rank", i)))

        # BM25 scoring on truncated doc text
        query_tokens = _default_tokenize(query)
        doc_texts = [_truncate(d.page_content or "", cfg.bm25_max_doc_chars) for d in docs]
        docs_tokens = [_default_tokenize(t) for t in doc_texts]
        bm25 = _bm25_scores(query_tokens, docs_tokens, k1=cfg.bm25_k1, b=cfg.bm25_b)

        # sparse ranks
        order_sparse = sorted(range(len(docs)), key=lambda i: bm25[i], reverse=True)
        sparse_ranks = [0] * len(docs)
        for r, i in enumerate(order_sparse, start=1):
            sparse_ranks[i] = r

        # attach sparse metadata
        for i, d in enumerate(docs):
            d.metadata["__bm25_score"] = float(bm25[i])
            d.metadata["__bm25_rank"] = int(sparse_ranks[i])

        fused = _rank_fusion(
            dense_ranks,
            sparse_ranks,
            mode=cfg.hybrid_fusion,
            w_dense=cfg.hybrid_dense_weight,
            w_sparse=cfg.hybrid_sparse_weight,
            rrf_k=cfg.rrf_k,
        )

        order = sorted(range(len(docs)), key=lambda i: fused[i], reverse=True)
        out: List[Document] = []
        for rank, i in enumerate(order, start=1):
            d = docs[i]
            d.metadata["__hybrid_score"] = float(fused[i])
            d.metadata["__hybrid_rank"] = int(rank)
            out.append(d)
        return out


    def triple_hybrid_retrieval(self, query: str) -> List[Document]:
        """
        Law/Rule/Case 3중 인덱스 검색 + 선택적 Rerank + 2-stage case 확장.
        Returns:
            최종 Document 리스트 (법적 위계 기반 컨텍스트에 바로 넣을 수 있는 형태)
        """
        cfg = self.config
        mult = cfg.search_multiplier

        logger.info(f"🔍 [통합 검색] query='{query}'")

        # 1) Retrieve (oversampling)
        docs_law = self._attach_source(
            self._search_dense_candidates(self.law_store, query, k=cfg.k_law * mult),
            "law",
        )
        docs_rule = self._attach_source(
            self._search_dense_candidates(self.rule_store, query, k=cfg.k_rule * mult),
            "rule",
        )
        # 2-stage: case는 청크 후보를 넉넉히 확보
        docs_case_chunks = self._attach_source(
            self._search_dense_candidates(self.case_store, query, k=cfg.case_candidate_k),
            "case",
        )

        # 1.5) Dense + Sparse(BM25) candidate-level hybrid re-ordering (per index)
        docs_law = self._dense_sparse_fuse(query, docs_law)
        docs_rule = self._dense_sparse_fuse(query, docs_rule)
        docs_case_chunks = self._dense_sparse_fuse(query, docs_case_chunks)

        # 2) Prepare rerank input (cap)
        combined_for_rerank = self._cap_for_rerank(docs_law, docs_rule, docs_case_chunks)

        # 3) Rerank (optional)
        selected_docs: List[Document]
        ranked = self._rerank(query, combined_for_rerank) if cfg.enable_rerank else None

        if ranked:
            # threshold filtering
            filtered = [(i, s) for (i, s) in ranked if s >= cfg.rerank_threshold]
            if not filtered:
                # fallback: take top few if threshold too strict
                desired = min(cfg.k_law + cfg.k_rule + cfg.k_case, len(ranked))
                filtered = ranked[:desired]

            # rerank order 유지
            selected_docs = [combined_for_rerank[i] for (i, _s) in filtered]

            logger.info(f"📌 Rerank selected={len(selected_docs)} (threshold={cfg.rerank_threshold})")
        else:
            # no rerank: keep retrieval order
            selected_docs = combined_for_rerank

        # 4) Deduplicate (again)
        selected_docs = _dedupe_docs(selected_docs, cfg.dedupe_key_fields)

        # 5) Select top docs per source (law/rule) and top cases per case_no (2-stage expansion)
        law_ranked = [d for d in selected_docs if d.metadata.get("__source_index") == "law"]
        rule_ranked = [d for d in selected_docs if d.metadata.get("__source_index") == "rule"]
        case_ranked_chunks = [d for d in selected_docs if d.metadata.get("__source_index") == "case"]

        final_law = law_ranked[: cfg.k_law]
        final_rule = rule_ranked[: cfg.k_rule]

        # case: choose unique case_no in order, then expand only for top N
        top_n = cfg.case_expand_top_n if cfg.case_expand_top_n is not None else cfg.k_case
        seen_case_no = set()
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
                # 전문 확장 실패 시 청크 그대로 사용
                expanded_cases.append(d)
                continue

            title = d.metadata.get("title") or d.metadata.get("case_name") or str(case_no)
            md = dict(d.metadata)
            md["__expanded"] = True
            # 보통 case priority는 9가 기대되지만, 원본 유지
            expanded_cases.append(
                Document(
                    page_content=f"[판례 전문: {title}]\n{full_text}",
                    metadata=md,
                )
            )

        # k_case 제한(안전)
        final_case = expanded_cases[: cfg.k_case]

        # 6) Priority sort (법적 위계)
        final_docs = final_law + final_rule + final_case
        final_docs = sorted(final_docs, key=lambda x: _safe_int((x.metadata or {}).get("priority", 99), 99))

        return final_docs

    # ----------------------------
    # Context formatting
    # ----------------------------
    @staticmethod
    def format_context_with_hierarchy(docs: List[Document]) -> str:
        """
        검색된 문서를 법적 위계(Priority)에 따라 섹션별로 재구성합니다.
        """
        section_1_law: List[str] = []   # Priority 1, 2, 4, 5
        section_2_rule: List[str] = []  # Priority 3, 6, 7, 8, 11
        section_3_case: List[str] = []  # 기타 (주로 판례)

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
            parts.append("## [SECTION 1: 핵심 법령 (최우선 법적 근거)]\n" + "\n\n".join(section_1_law))
        if section_2_rule:
            parts.append("## [SECTION 2: 관련 규정 및 절차 (세부 기준)]\n" + "\n\n".join(section_2_rule))
        if section_3_case:
            parts.append("## [SECTION 3: 판례 및 해석 사례 (적용 예시)]\n" + "\n\n".join(section_3_case))

        return "\n\n".join(parts).strip()

    # ----------------------------
    # Answer generation
    # ----------------------------
    def generate_answer(self, user_input: str, *, skip_normalization: bool = False) -> str:
        """
        최종 답변 생성:
        (1) 질문 표준화 (optional)
        (2) 3중 검색 + rerank + 2-stage 판례 확장
        (3) 위계 컨텍스트 구성
        (4) LLM 답변 생성
        """
        # 1) Normalize
        normalized_query = user_input if skip_normalization else self.normalize_query(user_input)
        if not skip_normalization:
            logger.info(f"🔄 표준화된 질문: {normalized_query}")

        # 2) Retrieve
        retrieved_docs = self.triple_hybrid_retrieval(normalized_query)
        if not retrieved_docs:
            return "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다."

        # 3) Context
        hierarchical_context = self.format_context_with_hierarchy(retrieved_docs)

        # 4) Generate
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        chain = prompt | self._generation_llm | StrOutputParser()

        logger.info("🤖 답변 생성 중...")
        try:
            return str(chain.invoke({"context": hierarchical_context, "question": normalized_query})).strip()
        except Exception as e:
            logger.warning(f"⚠️ 답변 생성 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."


__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "INDEX_NAMES",
    "KEYWORD_DICT",
    "NORMALIZATION_PROMPT",
    "SYSTEM_PROMPT",
]
