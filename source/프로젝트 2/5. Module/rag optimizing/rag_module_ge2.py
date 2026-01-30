"""
RAG LLM Pipeline for Django Web Application
주택임대차 RAG 시스템 - 통합 검색 및 답변 생성 모듈

[주요 기능]
1. 사용자 질문 표준화 (normalize_query)
2. 3중 인덱스(Law/Rule/Case) 통합 검색 (triple_hybrid_retrieval)
3. 법적 위계(Priority)에 따른 컨텍스트 재정렬 (format_context_with_hierarchy)
4. 최종 답변 생성 (generate_final_answer)
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone

# Reranking import (Optional)
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️ Warning: cohere library not installed. Reranking will be disabled.")

# ==========================================
# 0. 설정 및 상수 정의
# ==========================================
INDEX_NAMES = {
    "law": "law-index-final",
    "rule": "rule-index-final",
    "case": "case-index-final"
}

# 법률 용어 사전 (하드코딩된 딕셔너리 유지)
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

# ==========================================
# 1. 초기화 함수 (Django 앱 시작 시 호출)
# ==========================================
def initialize_vector_stores():
    """
    환경 변수를 로드하고 Pinecone VectorStore 객체들을 초기화하여 반환합니다.
    """
    load_dotenv(override=True)
    
    pc_api_key = os.getenv("PINECONE_API_KEY")
    if not pc_api_key:
        raise ValueError("❌ PINECONE_API_KEY가 환경 변수에 설정되지 않았습니다.")

    embedding = UpstageEmbeddings(model="solar-embedding-1-large-passage")
    
    print("🔗 Pinecone 인덱스 연결 중...")
    stores = {}
    for key, index_name in INDEX_NAMES.items():
        stores[key] = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding,
            pinecone_api_key=pc_api_key
        )
    
    print("✅ 모든 벡터 스토어 로드 완료!")
    return stores['law'], stores['rule'], stores['case']

# ==========================================
# 2. 내부 로직 함수들
# ==========================================

def normalize_query(user_query: str) -> str:
    """
    LLM을 사용하여 사용자 질문을 법률 용어로 표준화합니다.
    """
    llm = ChatOllama(model="exaone3.5:2.4b", temperature=0)
    
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
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        return chain.invoke({"dictionary": LEGAL_KEYWORD_MAP, "question": user_query}).strip()
    except Exception as e:
        print(f"⚠️ 전처리 실패 (원본 사용): {e}")
        return user_query

def get_full_case_context(case_no: str, case_store: PineconeVectorStore) -> str:
    """
    특정 사건번호의 판례 전문을 가져옵니다.
    """
    try:
        results = case_store.similarity_search(
            query="판례 전문 검색",  # Dummy query for API requirement
            k=50,
            filter={"case_no": {"$eq": case_no}}
        )
        sorted_docs = sorted(results, key=lambda x: x.metadata.get('chunk_id', ''))
        
        seen_chunks = set()
        unique_docs = []
        for doc in sorted_docs:
            cid = doc.metadata.get('chunk_id')
            if cid and cid not in seen_chunks:
                unique_docs.append(doc)
                seen_chunks.add(cid)
                
        return "\n".join([doc.page_content for doc in unique_docs])
    except Exception as e:
        print(f"⚠️ 판례 로딩 실패 ({case_no}): {e}")
        return ""

def triple_hybrid_retrieval(query, law_store, rule_store, case_store, k_law=3, k_rule=3, k_case=3):
    """
    Law, Rule, Case 인덱스에서 문서를 검색하고 Reranking을 수행합니다.
    """
    print(f"🔍 [통합 검색] 쿼리: '{query}'")
    
    # 1. 병렬 검색
    docs_law = law_store.similarity_search(query, k=k_law)
    docs_rule = rule_store.similarity_search(query, k=k_rule)
    docs_case_initial = case_store.similarity_search(query, k=k_case * 2)
    
    # 2. 판례 문맥 확장
    docs_case_expanded = []
    seen_cases = set()
    for doc in docs_case_initial:
        case_no = doc.metadata.get('case_no')
        if case_no and case_no not in seen_cases:
            full_text = get_full_case_context(case_no, case_store)
            if full_text:
                new_doc = doc
                new_doc.page_content = f"[판례 전문: {doc.metadata.get('title')}]\n{full_text}"
                docs_case_expanded.append(new_doc)
                seen_cases.add(case_no)
            if len(docs_case_expanded) >= k_case:
                break
                
    combined_docs = docs_law + docs_rule + docs_case_expanded
    
    # 3. Reranking (Cohere)
    if COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"):
        try:
            co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            docs_content = [d.page_content for d in combined_docs]
            rerank_results = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=docs_content,
                top_n=len(combined_docs)
            )
            
            filtered_docs = []
            print("📊 Rerank 점수 (Top):")
            for r in rerank_results.results:
                if r.relevance_score > 0.10: # Threshold
                    doc = combined_docs[r.index]
                    print(f" - [{r.relevance_score:.4f}] {doc.metadata.get('title')}")
                    filtered_docs.append(doc)
            return filtered_docs
        except Exception as e:
            print(f"⚠️ Rerank 실패: {e}")
            
    return combined_docs

def format_context_with_hierarchy(docs: List[Document]) -> str:
    """
    검색된 문서를 법적 위계(Priority)에 따라 섹션별로 재구성합니다.
    """
    # Priority 기준으로 오름차순 정렬 (낮을수록 상위 법령)
    sorted_docs = sorted(docs, key=lambda x: int(x.metadata.get('priority', 99)))
    
    section_1_law = []   # Priority 1, 2, 4, 5
    section_2_rule = []  # Priority 3, 6, 7, 8, 11
    section_3_case = []  # Priority 9
    
    for doc in sorted_docs:
        p = int(doc.metadata.get('priority', 99))
        src = doc.metadata.get('src_title', '자료')
        title = doc.metadata.get('title', '')
        entry = f"[{src}] {title}\n{doc.page_content}"
        
        if p in [1, 2, 4, 5]:
            section_1_law.append(entry)
        elif p in [3, 6, 7, 8, 11]:
            section_2_rule.append(entry)
        else:
            section_3_case.append(entry)
            
    formatted_text = ""
    if section_1_law:
        formatted_text += "## [SECTION 1: 핵심 법령 (최우선 법적 근거)]\n" + "\n\n".join(section_1_law) + "\n\n"
    if section_2_rule:
        formatted_text += "## [SECTION 2: 관련 규정 및 절차 (세부 기준)]\n" + "\n\n".join(section_2_rule) + "\n\n"
    if section_3_case:
        formatted_text += "## [SECTION 3: 판례 및 해석 사례 (적용 예시)]\n" + "\n\n".join(section_3_case) + "\n\n"
        
    return formatted_text

# ==========================================
# 3. 메인 인터페이스 함수
# ==========================================

def generate_final_answer(user_input: str, law_store, rule_store, case_store) -> str:
    """
    사용자 질문을 받아 RAG 파이프라인 전체를 실행하고 답변을 반환합니다.
    """
    # 1. 질문 표준화
    normalized_query = normalize_query(user_input)
    print(f"🔄 표준화된 질문: {normalized_query}")
    
    # 2. 통합 검색 및 위계 정렬
    retrieved_docs = triple_hybrid_retrieval(
        normalized_query, law_store, rule_store, case_store
    )
    
    if not retrieved_docs:
        return "죄송합니다. 관련 법령이나 판례를 찾을 수 없습니다."

    # 3. 위계 구조화된 컨텍스트 생성
    hierarchical_context = format_context_with_hierarchy(retrieved_docs)

    # 4. LLM 답변 생성
    system_prompt = """
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
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    llm = ChatOllama(model="exaone3.5:2.4b", temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    
    print("🤖 답변 생성 중...")
    return chain.invoke({"context": hierarchical_context, "question": normalized_query})

# ==========================================
# 테스트 실행 블록
# ==========================================
if __name__ == "__main__":
    print("🚀 RAG 파이프라인 테스트 시작...")
    try:
        # 1. 초기화
        law, rule, case = initialize_vector_stores()
        
        # 2. 질문 테스트
        test_query = "집주인이 실거주한다고 나가라고 하는데, 진짜인지 의심스러워요. 어떻게 확인하죠?"
        answer = generate_final_answer(test_query, law, rule, case)
        
        print("\n" + "="*50)
        print(answer)
        print("="*50)
        
    except Exception as e:
        print(f"🔥 에러 발생: {e}")