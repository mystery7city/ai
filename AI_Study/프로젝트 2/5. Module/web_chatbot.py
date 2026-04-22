"""
Legal RAG Chatbot - Web Interface
Premium Streamlit-based chatbot for Korean housing lease legal Q&A
"""
import streamlit as st
import os
import logging
from dotenv import load_dotenv

# 1. Load environment variables first
load_dotenv()

# 2. Configure logging to suppress verbose output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)

# 3. Import RAG module (now in same directory)
try:
    from rag_module import create_pipeline, RAGConfig
except ImportError as e:
    st.error(f"❌ RAG 모듈 로드 실패: {e}")
    st.info("rag_module.py 파일이 같은 폴더에 있는지 확인하세요.")
    st.stop()


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="법률 AI 상담",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Custom CSS for Premium Design
# =============================================================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* User message highlight */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.2);
    }
    
    /* Input box styling */
    .stChatInput textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Info cards */
    .info-card {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-online { background: #10b981; }
    .status-offline { background: #ef4444; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None


# =============================================================================
# Pipeline Initialization (Cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def init_pipeline():
    """Initialize RAG pipeline with error handling."""
    try:
        config = RAGConfig(
            temperature=0.1,
            enable_rerank=True,
            enable_bm25=True,
        )
        pipeline = create_pipeline(config=config)
        return pipeline, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    
    # Pipeline status
    st.markdown("---")
    st.markdown("### 🔌 시스템 상태")
    
    # Initialize pipeline on first load
    if st.session_state.pipeline is None and st.session_state.pipeline_error is None:
        with st.spinner("AI 시스템 초기화 중..."):
            pipeline, error = init_pipeline()
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_error = error
    
    # Show status
    if st.session_state.pipeline:
        st.markdown('<span class="status-dot status-online"></span> **연결됨**', unsafe_allow_html=True)
        st.success("RAG 파이프라인 준비 완료")
    else:
        st.markdown('<span class="status-dot status-offline"></span> **오프라인**', unsafe_allow_html=True)
        st.error(f"초기화 실패: {st.session_state.pipeline_error}")
    
    # Clear chat button
    st.markdown("---")
    st.markdown("### 💬 대화")
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Info section
    st.markdown("---")
    st.markdown("### ℹ️ 도움말")
    with st.expander("사용 방법"):
        st.markdown("""
        1. 하단 입력창에 질문을 입력하세요
        2. 주택 임대차, 전월세 관련 법률 질문이 최적입니다
        3. AI가 관련 법령, 규정, 판례를 검색하여 답변합니다
        """)
    
    with st.expander("예시 질문"):
        st.markdown("""
        - 전세 보증금 반환 절차는?
        - 묵시적 갱신이란 무엇인가요?
        - 집주인이 수리를 해주지 않으면?
        - 전세 사기 예방 방법은?
        - 계약 갱신 청구권 사용 조건은?
        """)
    
    # Footer
    st.markdown("---")
    st.caption("⚖️ 법률 AI 상담 v1.0")
    st.caption("📚 주택임대차보호법 기반")


# =============================================================================
# Main Chat Interface
# =============================================================================
st.markdown("# ⚖️ 법률 AI 상담")
st.markdown("주택 임대차 · 전월세 전문 법률 상담 AI")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "⚖️"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("법률 관련 질문을 입력하세요..."):
    # Check if pipeline is ready
    if not st.session_state.pipeline:
        st.error("❌ AI 시스템이 준비되지 않았습니다. 페이지를 새로고침 해주세요.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar="⚖️"):
        message_placeholder = st.empty()
        
        with st.spinner("🔍 법령 및 판례 검색 중..."):
            try:
                response = st.session_state.pipeline.generate_answer(prompt)
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Empty state message
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.6);">
        <h3>👋 안녕하세요!</h3>
        <p>주택 임대차 관련 법률 질문을 해주세요.</p>
        <p style="font-size: 0.9rem;">예: "전세 계약 만료 시 보증금은 어떻게 돌려받나요?"</p>
    </div>
    """, unsafe_allow_html=True)
