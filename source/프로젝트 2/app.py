import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

st.set_page_config(page_title="전세/임대차 RAG", layout="wide")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"

RAG_RULES = """
너는 오직 제공된 [근거]만으로 답한다.
- 근거에 없는 사실/법리/절차/판례/조문을 절대 추가하지 마라.
- 추론이 필요하면 '근거 부족'이라고 말하고, 무엇이 부족한지 적어라.
- 반드시 근거에서 원문 문장(따옴표) 2~3개를 인용해라.
- 결론은 1~2문장으로 짧게.
"""

def embed_query(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def search_docs(query: str, top_k=8, min_score=0.45):
    v = embed_query(query)
    res = index.query(vector=v, top_k=top_k, include_metadata=True)
    return [m for m in res["matches"] if m["score"] >= min_score]

def answer_with_rag(query: str):
    matches = search_docs(query)
    if not matches:
        return "근거 부족: 관련 문서를 찾지 못했습니다.", []

    context = "\n\n".join(
        f"[근거 {i+1} | score={m['score']:.3f}] {m['metadata'].get('text','')}"
        for i, m in enumerate(matches)
    )

    prompt = f"""{RAG_RULES}

[근거]
{context}

[질문]
{query}

[출력 형식]
1) 결론
2) 근거 인용(따옴표 2~3개)
3) 요약(3줄 이내)
4) 근거 부족한 부분(있으면)
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content, matches

st.title("전세/임대차 문서 RAG")
q = st.text_input("질문을 입력하세요")

if st.button("질문하기") and q.strip():
    ans, matches = answer_with_rag(q)
    st.subheader("답변")
    st.write(ans)
    st.subheader("검색 근거")
    for m in matches:
        st.write(m["metadata"].get("text",""))
        st.divider()
