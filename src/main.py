"""
main.py - AI Chatbot Platform — Streamlit Dashboard

Integrates all 6 internship feature modules with a clean, minimal UI:
1. 🏠 Customer Service FAQs & Dynamic Knowledge Base
2. 🎨 Multi-Modal Hub (Image Analysis & Visual Q&A)
3. 🩺 MedQuAD Medical Q&A
4. 🔬 arXiv Domain Expert
5. 📊 Sentiment Analytics & Emotion-Adaptive Response
6. 🌐 Multilingual Settings

Built on top of the original customer_service_chatbot_LLM training project.
"""

import os
import sys
import json
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Chatbot Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Minimal & Theme-Friendly
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* === Metric Cards === */
    .metric-card {
        background: rgba(102, 126, 234, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.25);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .metric-card .metric-value {
        font-size: 2rem; font-weight: 700; color: #667eea !important;
    }
    .metric-card .metric-label {
        font-size: 0.85rem; opacity: 0.8; margin-top: 0.3rem;
    }

    /* === Sentiment Colors === */
    .sentiment-positive {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white !important; padding: 0.8rem; border-radius: 10px;
        font-size: 1rem; font-weight: 600; text-align: center;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white !important; padding: 0.8rem; border-radius: 10px;
        font-size: 1rem; font-weight: 600; text-align: center;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important; padding: 0.8rem; border-radius: 10px;
        font-size: 1rem; font-weight: 600; text-align: center;
    }
    .sentiment-urgent {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #222 !important; padding: 0.8rem; border-radius: 10px;
        font-size: 1rem; font-weight: 600; text-align: center;
    }

    /* === Entity Tags === */
    .entity-tag {
        display: inline-block; padding: 0.2rem 0.6rem;
        border-radius: 6px; font-size: 0.8rem; font-weight: 600;
        margin: 0.15rem; border: 1px solid;
    }
    .entity-symptoms { background: rgba(255, 193, 7, 0.18); color: #e5a100 !important; border-color: #ffc107; }
    .entity-diseases { background: rgba(235, 51, 73, 0.18); color: #f45c43 !important; border-color: #eb3349; }
    .entity-treatments { background: rgba(56, 239, 125, 0.18); color: #28c76f !important; border-color: #38ef7d; }
    .entity-body_parts { background: rgba(0, 180, 216, 0.18); color: #00b4d8 !important; border-color: #00b4d8; }

    /* === Answer Box === */
    .answer-box {
        background: rgba(102, 126, 234, 0.08);
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        border-top: 1px solid rgba(102, 126, 234, 0.15);
        border-right: 1px solid rgba(102, 126, 234, 0.15);
        border-bottom: 1px solid rgba(102, 126, 234, 0.15);
    }

    /* === Confidence Meter === */
    .confidence-bar {
        height: 8px; border-radius: 4px; overflow: hidden;
        background: rgba(128, 128, 128, 0.2); margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.5s ease;
    }
    .conf-high { background: linear-gradient(90deg, #11998e, #38ef7d); }
    .conf-medium { background: linear-gradient(90deg, #f7971e, #ffd200); }
    .conf-low { background: linear-gradient(90deg, #eb3349, #f45c43); }

    /* === Status Badges === */
    .badge-active {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white !important; padding: 0.25rem 0.8rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; display: inline-block;
    }
    .badge-inactive {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white !important; padding: 0.25rem 0.8rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; display: inline-block;
    }
    .badge-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important; padding: 0.25rem 0.8rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; display: inline-block;
    }

    /* === Paper Card === */
    .paper-card {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 12px; padding: 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 0.8rem;
    }
    .paper-title {
        font-weight: 600; font-size: 1rem; margin-bottom: 0.4rem;
    }
    .paper-meta {
        font-size: 0.8rem; opacity: 0.75;
    }

    /* === Feature Card === */
    .feature-card {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 12px; padding: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
    }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e8e8e8 !important;
    }

    /* === Hide default streamlit elements === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Import project modules (with graceful fallbacks)
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from langchain_helper import create_vector_db, get_qa_chain, create_domain_vector_db, get_llm_response
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False

try:
    from knowledge_manager import KnowledgeBaseManager
    KNOWLEDGE_MANAGER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_MANAGER_AVAILABLE = False

try:
    from multimodal_engine import MultiModalEngine
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

try:
    from medical_qa_engine import MedicalQAEngine
    MEDICAL_QA_AVAILABLE = True
except ImportError:
    MEDICAL_QA_AVAILABLE = False

try:
    from arxiv_expert_engine import ArXivExpertEngine
    ARXIV_EXPERT_AVAILABLE = True
except ImportError:
    ARXIV_EXPERT_AVAILABLE = False

try:
    from sentiment_engine import SentimentEngine
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from multilingual_engine import MultilingualEngine
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cached Engine Initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def get_kb_manager():
    if KNOWLEDGE_MANAGER_AVAILABLE:
        return KnowledgeBaseManager(
            vectordb_path=os.path.join(PROJECT_DIR, "faiss_index"),
            metadata_path=os.path.join(PROJECT_DIR, "kb_metadata.json"),
        )
    return None

def get_multimodal_engine():
    if MULTIMODAL_AVAILABLE:
        return MultiModalEngine()
    return None

@st.cache_resource
def get_medical_engine():
    if MEDICAL_QA_AVAILABLE:
        return MedicalQAEngine(
            dataset_path=os.path.join(PROJECT_DIR, "dataset", "medquad_dataset.csv"),
            vectordb_path=os.path.join(PROJECT_DIR, "medical_faiss_index"),
        )
    return None

@st.cache_resource
def get_arxiv_engine():
    if ARXIV_EXPERT_AVAILABLE:
        return ArXivExpertEngine(
            dataset_path=os.path.join(PROJECT_DIR, "dataset", "arxiv_dataset.json"),
            vectordb_path=os.path.join(PROJECT_DIR, "arxiv_faiss_index"),
        )
    return None

@st.cache_resource
def get_sentiment_engine():
    if SENTIMENT_AVAILABLE:
        return SentimentEngine()
    return None

@st.cache_resource
def get_multilingual_engine():
    if MULTILINGUAL_AVAILABLE:
        return MultilingualEngine()
    return None


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "chat_history": [],
        "medical_chat_history": [],
        "arxiv_chat_history": [],
        "multilingual_chat_history": [],
        "ml_selected_chip": "",
        "current_language": "en",
        "sentiment_active": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Load cached engines
kb_manager = get_kb_manager()
multimodal_engine = get_multimodal_engine()
medical_engine = get_medical_engine()
arxiv_engine = get_arxiv_engine()
sentiment_engine = get_sentiment_engine()
multilingual_engine = get_multilingual_engine()


# ---------------------------------------------------------------------------
# Helper: Render styled components
# ---------------------------------------------------------------------------

def render_metric_card(value, label, col):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_entity_tags(entities: dict):
    html = ""
    for etype, elist in entities.items():
        for e in elist:
            html += f'<span class="entity-tag entity-{etype}">{etype.upper()}: {e}</span> '
    if html:
        st.markdown(html, unsafe_allow_html=True)

def render_confidence(score: float):
    pct = int(score * 100)
    cls = "conf-high" if score > 0.7 else "conf-medium" if score > 0.4 else "conf-low"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-weight:600;font-size:0.9rem;">Confidence: {pct}%</span>
        <div class="confidence-bar" style="flex:1;">
            <div class="confidence-fill {cls}" style="width:{pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_answer_box(text: str):
    st.markdown(f'<div class="answer-box">{text}</div>', unsafe_allow_html=True)

def render_sentiment_card(sentiment: str, score: float, emoji: str):
    cls_map = {"positive": "positive", "satisfied": "positive",
               "negative": "negative", "frustrated": "negative",
               "urgent": "urgent", "neutral": "neutral"}
    cls = cls_map.get(sentiment, "neutral")
    st.markdown(f"""
    <div class="sentiment-{cls}">
        {emoji} {sentiment.upper()} &nbsp;|&nbsp; Score: {score:.2f}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Clean & Compact
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:3rem;">🤖</div>
        <h2 style="margin:0.3rem 0;font-size:1.3rem;">AI Chatbot Platform</h2>
        <p style="opacity:0.7;font-size:0.8rem;margin:0;">Powered by Google Gemini & LangChain</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Language Selector
    if multilingual_engine:
        langs = multilingual_engine.get_supported_languages()
        lang_options = {f"{l['flag']} {l['name']}": l["code"] for l in langs}
        selected_lang = st.selectbox("🌐 Language", list(lang_options.keys()), index=0)
        st.session_state.current_language = lang_options[selected_lang]
        multilingual_engine.set_language(st.session_state.current_language)

    st.session_state.sentiment_active = st.toggle("📊 Sentiment Analysis", value=True)

    st.divider()

    # Module Status — Hidden by default
    with st.expander("📡 Module Status"):
        modules_status = {
            "FAQ Knowledge Base": LANGCHAIN_AVAILABLE,
            "Dynamic KB Manager": KNOWLEDGE_MANAGER_AVAILABLE,
            "Multi-Modal Engine": MULTIMODAL_AVAILABLE,
            "Medical Q&A": MEDICAL_QA_AVAILABLE,
            "arXiv Expert": ARXIV_EXPERT_AVAILABLE,
            "Sentiment Analysis": SENTIMENT_AVAILABLE,
            "Multilingual": MULTILINGUAL_AVAILABLE,
        }
        for name, available in modules_status.items():
            badge = "active" if available else "inactive"
            label = "Online" if available else "Offline"
            st.markdown(f'<span class="badge-{badge}">{label}</span> {name}', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Customer Service",
    "🎨 Multi-Modal",
    "🩺 Medical Q&A",
    "🔬 arXiv Expert",
    "📊 Sentiment",
    "🌐 Multilingual",
])


# ============================== TAB 1 ==============================
with tab1:
    st.markdown("### 💬 Ask About Nullclass Courses & Services")

    question = st.text_input("💭 Your question:", key="faq_question",
                             placeholder="e.g., Do you offer EMI payments? Do you have a JavaScript course?")

    if question:
        processed_q = question
        detected_lang = "en"
        target_reply_lang = "en"

        if multilingual_engine:
            ml_result = multilingual_engine.process_multilingual_query(question)
            detected_info = ml_result.get("detected_language", {})
            detected_lang = detected_info.get("language_code", "en")
            if detected_lang != "en" or st.session_state.current_language != "en":
                processed_q = ml_result.get("english_text", question)
                target_reply_lang = detected_lang if detected_lang != "en" else st.session_state.current_language
                st.caption(f"🌐 Auto-detected: **{detected_info.get('flag', '')} {detected_info.get('language_name', 'Language')}** (translated to English for search)")

        if st.session_state.sentiment_active and sentiment_engine:
            sent = sentiment_engine.analyze_sentiment(question)
            render_sentiment_card(sent.get("sentiment", "neutral"), sent.get("score", 0), sent.get("emoji_indicator", ""))

        chain = get_qa_chain()
        if chain:
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    response = chain({"query": processed_q})
                    answer = response.get("result", "I don't know.")

                    if target_reply_lang != "en" and multilingual_engine:
                        ml_resp = multilingual_engine.format_multilingual_response(answer, target_reply_lang)
                        answer = ml_resp.get("response_text", answer)

                    render_answer_box(answer)

                    with st.expander("📄 Source Documents"):
                        for i, doc in enumerate(response.get("source_documents", [])):
                            st.caption(f"**Source {i+1}:** {doc.page_content[:200]}...")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("⚠️ Click 'Create Knowledge Base' in the admin panel below first.")

    # Admin Tools — Hidden by default
    st.divider()
    with st.expander("⚙️ Admin: Knowledge Base Management"):
        if st.button("🔧 Create / Rebuild Knowledge Base", key="create_kb", use_container_width=True):
            with st.spinner("Building knowledge base from FAQ dataset..."):
                csv_path = os.path.join(PROJECT_DIR, "dataset", "dataset.csv")
                success = create_vector_db(csv_path=csv_path)
                if success:
                    st.success("✅ Knowledge base created successfully!")
                else:
                    st.error("❌ Failed to create knowledge base.")

        if kb_manager:
            status = kb_manager.get_status()
            c1, c2 = st.columns(2)
            render_metric_card(status.get("document_count", 0), "Documents", c1)
            render_metric_card(f"v{status.get('version', 0)}", "Version", c2)
            st.caption(f"🕐 Last updated: {status.get('last_updated', 'Never')}")

            st.divider()
            upload_type = st.selectbox("Add source:", ["CSV File", "Text Input", "URL"], key="kb_type")

            if upload_type == "CSV File":
                f = st.file_uploader("Upload CSV", type=["csv"], key="kb_csv")
                if f and st.button("📥 Ingest", key="ingest_csv"):
                    temp = os.path.join(PROJECT_DIR, "temp_upload.csv")
                    with open(temp, "wb") as fp:
                        fp.write(f.getvalue())
                    with st.spinner("Ingesting..."):
                        r = kb_manager.add_csv_source(temp)
                        st.success(f"✅ Added {r.get('added_docs', 0)} docs!")
                    os.remove(temp)
            elif upload_type == "Text Input":
                txt = st.text_area("Enter knowledge:", key="kb_text", height=100)
                if txt and st.button("📥 Add", key="ingest_text"):
                    r = kb_manager.add_text_documents([txt])
                    st.success(f"✅ Added {r.get('added_docs', 0)} docs!")
            elif upload_type == "URL":
                url = st.text_input("URL:", key="kb_url")
                if url and st.button("📥 Fetch", key="ingest_url"):
                    with st.spinner("Fetching..."):
                        r = kb_manager.add_url_source(url)
                        st.success(f"✅ Added {r.get('added_docs', 0)} docs!")

            if st.button("🔄 Check Updates", key="check_updates", use_container_width=True):
                sources = [os.path.join(PROJECT_DIR, "dataset", "dataset.csv")]
                changed = kb_manager.check_for_updates(sources)
                if changed:
                    st.warning(f"📢 {len(changed)} source(s) changed!")
                else:
                    st.success("✅ All sources up to date.")
        else:
            st.info("KB Manager not available.")


# ============================== TAB 2 ==============================
with tab2:
    st.markdown("### 🎨 Multi-Modal AI Hub")
    st.caption("Upload images for AI-powered analysis, visual Q&A, text extraction, and comparison.")

    if MULTIMODAL_AVAILABLE and multimodal_engine:
        engine_status = multimodal_engine.get_status()

        if engine_status.get("available", False):
            col1, col2 = st.columns([1, 1])

            with col1:
                uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png", "webp"], key="mm_image")
                if uploaded_image:
                    from PIL import Image
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                if uploaded_image:
                    from PIL import Image
                    image = Image.open(uploaded_image)

                    mode = st.radio("Analysis Mode:", [
                        "🔍 Describe Image", "❓ Visual Q&A",
                        "📝 Extract Text (OCR)", "🏷️ Generate Metadata"
                    ], key="mm_mode", horizontal=True)

                    if mode == "🔍 Describe Image":
                        prompt = st.text_input("Custom prompt (optional):", value="Describe this image in detail.", key="mm_prompt")
                        if st.button("🚀 Analyze", key="mm_go", use_container_width=True):
                            with st.spinner("Analyzing image..."):
                                result = multimodal_engine.analyze_image(image, prompt)
                                render_answer_box(result)

                    elif mode == "❓ Visual Q&A":
                        vq = st.text_input("Ask about the image:", key="mm_vqa")
                        if vq and st.button("🚀 Get Answer", key="mm_vqa_go", use_container_width=True):
                            with st.spinner("Processing..."):
                                render_answer_box(multimodal_engine.visual_qa(image, vq))

                    elif mode == "📝 Extract Text (OCR)":
                        if st.button("🚀 Extract Text", key="mm_ocr", use_container_width=True):
                            with st.spinner("Extracting..."):
                                st.code(multimodal_engine.extract_text_from_image(image))

                    elif mode == "🏷️ Generate Metadata":
                        if st.button("🚀 Generate", key="mm_meta", use_container_width=True):
                            with st.spinner("Generating..."):
                                st.json(multimodal_engine.generate_image_description_for_search(image))

            # Image Comparison & Generation — Hidden by default in expanders
            st.divider()
            with st.expander("🔄 Image Comparison"):
                cc1, cc2 = st.columns(2)
                img1_f = cc1.file_uploader("Image 1", type=["jpg","jpeg","png"], key="ci1")
                img2_f = cc2.file_uploader("Image 2", type=["jpg","jpeg","png"], key="ci2")
                if img1_f and img2_f:
                    from PIL import Image
                    i1, i2 = Image.open(img1_f), Image.open(img2_f)
                    cc1.image(i1, caption="Image 1", use_container_width=True)
                    cc2.image(i2, caption="Image 2", use_container_width=True)
                    if st.button("🔍 Compare", key="compare", use_container_width=True):
                        with st.spinner("Comparing..."):
                            render_answer_box(multimodal_engine.compare_images(i1, i2))

            with st.expander("🎨 AI Image Generation"):
                st.caption("Generate safe, photorealistic, and contextually accurate visual content using Generative AI.")
                img_col1, img_col2 = st.columns([3, 1])
                with img_col1:
                    gen_prompt = st.text_input(
                        "Describe the image to generate:",
                        placeholder="e.g., A friendly AI customer support specialist assisting students with programming courses",
                        key="gen_img_prompt"
                    )
                with img_col2:
                    img_style = st.selectbox(
                        "Style:",
                        ["photorealistic", "3d_render", "digital_art", "illustration"],
                        index=0,
                        key="gen_img_style"
                    )


                if gen_prompt and st.button("✨ Generate Image", key="btn_gen_img", use_container_width=True):
                    with st.spinner("Synthesizing context-accurate image with Generative AI..."):
                        # Direct call on engine or fresh instance
                        from multimodal_engine import MultiModalEngine
                        current_engine = multimodal_engine if isinstance(multimodal_engine, MultiModalEngine) else MultiModalEngine()
                        gen_img = current_engine.generate_image(gen_prompt, style=img_style)
                        if gen_img:
                            st.image(gen_img, caption=f"Generated ({img_style}): {gen_prompt}", use_container_width=True)
                            st.success("✅ Image generated successfully with safety and quality guardrails!")
                        else:
                            st.error("❌ Failed to generate image. Please ensure prompt is safe and try again.")



        else:
            st.warning(f"⚠️ Multi-Modal Engine unavailable. {engine_status.get('error', 'Check API key.')}")
    else:
        st.info("Multi-Modal Engine not loaded. Install `google-generativeai` and set `GOOGLE_API_KEY`.")


# ============================== TAB 3 ==============================
with tab3:
    st.markdown("### 🩺 Medical Q&A Chatbot")

    st.warning("⚠️ **Medical Disclaimer:** This provides general health information only. Always consult a qualified healthcare provider for medical advice.")

    if MEDICAL_QA_AVAILABLE and medical_engine:
        med_q = st.text_input("🩺 Ask a medical question:", key="med_q",
                              placeholder="e.g., What are the symptoms of diabetes?")

        if med_q:
            with st.spinner("🔍 Searching medical knowledge base..."):
                result = medical_engine.get_answer(med_q)

                # Entity tags
                entities = result.get("entities", {})
                if entities:
                    st.markdown("**🏷️ Detected Medical Entities:**")
                    render_entity_tags(entities)

                # Answer
                st.markdown("")
                render_answer_box(result.get("answer", "No answer found."))

                # Confidence
                render_confidence(result.get("confidence", 0))

                # Category
                if result.get("category"):
                    st.markdown(f'<span class="badge-info">📁 {result["category"]}</span>', unsafe_allow_html=True)

                # Disclaimer
                st.caption(f"ℹ️ {result.get('disclaimer', medical_engine.get_safety_disclaimer())[:150]}...")

                st.session_state.medical_chat_history.append({
                    "question": med_q, "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0),
                })

        # Admin Tools — Hidden by default
        st.divider()
        with st.expander("⚙️ Medical KB Management & History"):
            if st.button("🔧 Build Medical Knowledge Base", key="build_med", use_container_width=True):
                with st.spinner("Building medical KB..."):
                    success = medical_engine.create_medical_vector_db()
                    st.success("✅ Medical KB ready!" if success else "Using keyword fallback.")

            if medical_engine:
                ms = medical_engine.get_status()
                c1, c2 = st.columns(2)
                render_metric_card(ms.get("dataset_size", 0), "Q&A Pairs", c1)
                vr = "✅" if ms.get("vector_store_ready") else "❌"
                render_metric_card(vr, "Vector Store", c2)

                categories = ms.get("categories", [])
                if categories:
                    st.markdown("**Browse by Category:**")
                    sel_cat = st.selectbox("Category:", categories, key="med_cat")
                    if sel_cat:
                        results = medical_engine.search_by_category(sel_cat)
                        for item in results[:5]:
                            with st.expander(f"💬 {item.get('question', '')[:55]}..."):
                                st.write(item.get("answer", ""))

            if st.session_state.medical_chat_history:
                st.divider()
                st.markdown("**📜 Recent Questions**")
                for e in reversed(st.session_state.medical_chat_history[-3:]):
                    st.caption(f"Q: {e['question'][:45]}... | Conf: {e['confidence']:.0%}")
    else:
        st.info("Medical Q&A not loaded. Ensure `medical_qa_engine.py` and `medquad_dataset.csv` are present.")


# ============================== TAB 4 ==============================
with tab4:
    st.markdown("### 🔬 arXiv Domain Expert — CS / AI Research")
    st.caption("Search, summarize, and explore cutting-edge research papers.")

    if ARXIV_EXPERT_AVAILABLE and arxiv_engine:
        if st.button("🔧 Build Paper Index", key="build_arxiv", use_container_width=True):
            with st.spinner("Indexing papers..."):
                success = arxiv_engine.create_paper_vector_db()
                st.success("✅ Paper index built!" if success else "Using keyword search.")

        mode = st.radio("Mode:", [
            "🔍 Search", "📝 Summarize", "💡 Explain",
            "❓ Ask", "📊 Statistics", "🕸️ Concept Map"
        ], horizontal=True, key="arxiv_mode")

        st.divider()

        if mode == "🔍 Search":
            q = st.text_input("🔍 Search query:", key="ax_search", placeholder="e.g., transformer attention mechanism")
            if q:
                with st.spinner("Searching..."):
                    results = arxiv_engine.search_papers(q, top_k=5)
                    for p in results:
                        st.markdown(f"""
                        <div class="paper-card">
                            <div class="paper-title">📄 {p.get('title', 'Untitled')}</div>
                            <div class="paper-meta">
                                👥 {', '.join(p.get('authors', [])[:3])} &nbsp;|&nbsp;
                                📅 {p.get('year', '')} &nbsp;|&nbsp;
                                🏷️ {', '.join(p.get('categories', [])[:3])}
                            </div>
                            <p style="font-size:0.9rem;margin-top:0.5rem;">{p.get('abstract', '')[:200]}...</p>
                        </div>
                        """, unsafe_allow_html=True)

        elif mode == "📝 Summarize":
            papers = arxiv_engine._load_dataset()
            opts = {f"{p.get('title','')[:60]} ({p.get('id','')})": p.get("id","") for p in papers}
            sel = st.selectbox("Select paper:", list(opts.keys()), key="ax_paper")
            level = st.selectbox("Level:", ["executive", "technical", "layperson"], key="ax_lvl")
            if sel and st.button("📝 Generate Summary", key="gen_sum", use_container_width=True):
                with st.spinner("Summarizing..."):
                    render_answer_box(arxiv_engine.summarize_paper(opts[sel], level))

        elif mode == "💡 Explain":
            concept = st.text_input("Concept:", key="ax_concept", placeholder="e.g., attention mechanism, GAN")
            if concept and st.button("💡 Explain", key="explain", use_container_width=True):
                with st.spinner("Generating explanation..."):
                    render_answer_box(arxiv_engine.explain_concept(concept))

        elif mode == "❓ Ask":
            q = st.text_input("Ask about CS/AI:", key="ax_q", placeholder="e.g., What are advances in NLP?")
            if q:
                with st.spinner("Researching..."):
                    r = arxiv_engine.answer_question(q, [h.get("question","") for h in st.session_state.arxiv_chat_history[-3:]])
                    render_answer_box(r.get("answer", ""))
                    if r.get("sources"):
                        with st.expander("📚 Sources"):
                            for s in r["sources"]:
                                st.caption(f"• {s.get('title', 'Unknown')}")
                    st.session_state.arxiv_chat_history.append({"question": q, "answer": r.get("answer","")})

        elif mode == "📊 Statistics":
            stats = arxiv_engine.get_field_statistics()
            c1, c2, c3 = st.columns(3)
            render_metric_card(stats.get("total_papers", 0), "Total Papers", c1)
            render_metric_card(len(stats.get("category_distribution", {})), "Categories", c2)
            render_metric_card(len(stats.get("year_distribution", {})), "Year Range", c3)

            col_l, col_r = st.columns(2)
            with col_l:
                if stats.get("category_distribution"):
                    import plotly.express as px
                    cd = stats["category_distribution"]
                    fig = px.pie(names=list(cd.keys()), values=list(cd.values()), title="Category Distribution",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=300)
                    st.plotly_chart(fig, use_container_width=True)
            with col_r:
                if stats.get("year_distribution"):
                    import plotly.express as px
                    yd = stats["year_distribution"]
                    fig2 = px.bar(x=list(yd.keys()), y=list(yd.values()), title="Papers by Year",
                                 labels={"x":"Year","y":"Count"}, color_discrete_sequence=["#667eea"])
                    fig2.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=300)
                    st.plotly_chart(fig2, use_container_width=True)

            trends = arxiv_engine.get_trending_topics(top_n=8)
            if trends:
                st.markdown("**🔥 Trending Topics:**")
                cols = st.columns(4)
                for i, (term, count) in enumerate(trends):
                    cols[i % 4].markdown(f'<span class="badge-info">{term} ({count})</span>', unsafe_allow_html=True)

        elif mode == "🕸️ Concept Map":
            mq = st.text_input("Topic:", key="cmap", placeholder="e.g., deep learning")
            if mq and st.button("🕸️ Generate", key="gen_map", use_container_width=True):
                with st.spinner("Generating concept map..."):
                    gd = arxiv_engine.generate_concept_graph_data(mq)
                    nodes, edges = gd.get("nodes",[]), gd.get("edges",[])
                    if nodes:
                        import plotly.graph_objects as go
                        import math
                        n = len(nodes)
                        pos = {nd["id"]: (math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n)) for i, nd in enumerate(nodes)}
                        ex, ey = [], []
                        for e in edges:
                            if e["source"] in pos and e["target"] in pos:
                                ex.extend([pos[e["source"]][0], pos[e["target"]][0], None])
                                ey.extend([pos[e["source"]][1], pos[e["target"]][1], None])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1,color="#ccc"),hoverinfo="none"))
                        fig.add_trace(go.Scatter(
                            x=[pos[nd["id"]][0] for nd in nodes], y=[pos[nd["id"]][1] for nd in nodes],
                            mode="markers+text", marker=dict(size=[nd.get("size",10)*3 for nd in nodes],
                            color="#667eea",line=dict(width=2,color="white")),
                            text=[nd["label"] for nd in nodes], textposition="top center", hoverinfo="text"
                        ))
                        fig.update_layout(showlegend=False,margin=dict(l=10,r=10,t=10,b=10),height=450,
                                         plot_bgcolor="white")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No concepts found for this query.")
    else:
        st.info("arXiv Expert not loaded. Ensure `arxiv_expert_engine.py` and `arxiv_dataset.json` are present.")


# ============================== TAB 5 ==============================
with tab5:
    st.markdown("### 📊 Sentiment Analytics & Emotion-Adaptive Response")
    st.caption("Analyze customer emotions and see adaptive response recommendations.")

    if SENTIMENT_AVAILABLE and sentiment_engine:
        text_input = st.text_area("💬 Enter text to analyze:", key="sent_text", height=100,
            placeholder="e.g., I'm really frustrated with the service! This is unacceptable!")

        if text_input and st.button("🔍 Analyze Sentiment", key="analyze_sent", use_container_width=True):
            result = sentiment_engine.analyze_sentiment(text_input)
            emoji = result.get("emoji_indicator", "")
            sentiment = result.get("sentiment", "neutral")
            score = result.get("score", 0)
            intensity = result.get("intensity", "mild")

            # Sentiment Card
            render_sentiment_card(sentiment, score, emoji)
            st.markdown(f"**Intensity:** `{intensity}` &nbsp;|&nbsp; **Urgent:** `{result.get('is_urgent', False)}`")

            # Matched words
            c1, c2 = st.columns(2)
            if result.get("matched_positive"):
                c1.success(f"✅ Positive: {', '.join(result['matched_positive'])}")
            if result.get("matched_negative"):
                c2.error(f"❌ Negative: {', '.join(result['matched_negative'])}")

            # Response modifier
            st.divider()
            st.markdown("**🎯 Adaptive Response Recommendation:**")
            mod = sentiment_engine.get_response_modifier(result)

            st.markdown(f"""
            <div class="feature-card">
                <p><strong>Recommended Tone:</strong> {mod.get('tone', 'professional')}</p>
                <p><strong>Suggested Opening:</strong> <em>{mod.get('prefix_suggestion', 'N/A')}</em></p>
                <p><strong>Suggested Closing:</strong> <em>{mod.get('suffix_suggestion', 'N/A')}</em></p>
                {'<p style="color:#eb3349 !important;font-weight:600;">⬆️ ESCALATION RECOMMENDED</p>' if mod.get('escalate') else ''}
            </div>
            """, unsafe_allow_html=True)

        # Session Analytics — Hidden by default
        st.divider()
        with st.expander("📈 Session Analytics"):
            analytics = sentiment_engine.get_session_analytics()

            c1, c2 = st.columns(2)
            render_metric_card(analytics.get("total_messages", 0), "Messages", c1)
            render_metric_card(f"{analytics.get('avg_score', 0):.2f}", "Avg Score", c2)

            c3, c4 = st.columns(2)
            render_metric_card(f"{analytics.get('satisfaction_rate', 0):.0%}", "Satisfaction", c3)
            trend = analytics.get("trend", "stable")
            trend_emoji = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
            render_metric_card(f"{trend_emoji}", trend.title(), c4)

            # Distribution pie
            dist = analytics.get("distribution", {})
            if dist and sum(dist.values()) > 0:
                import plotly.express as px
                fig = px.pie(names=list(dist.keys()), values=list(dist.values()),
                            color_discrete_sequence=["#38ef7d","#eb3349","#667eea","#f7971e","#11998e","#ffd200"])
                fig.update_layout(margin=dict(l=5,r=5,t=5,b=5), height=200, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            # Trend line
            trend_data = sentiment_engine.get_sentiment_trend()
            if trend_data:
                import plotly.express as px
                scores = [t["score"] for t in trend_data]
                fig2 = px.line(x=list(range(len(scores))), y=scores, labels={"x":"Message #","y":"Score"})
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                fig2.update_layout(margin=dict(l=5,r=5,t=5,b=5), height=180)
                st.plotly_chart(fig2, use_container_width=True)

            if st.button("🔄 Reset Session", key="reset_sent", use_container_width=True):
                sentiment_engine.reset_session()
                st.success("Session reset!")
                st.rerun()
    else:
        st.info("Sentiment Engine not loaded.")


# ============================== TAB 6 ==============================
with tab6:
    st.markdown("### 🌐 Multilingual Customer Service AI")
    st.caption("Chat in English, Spanish, French, German, or Hindi — automatic language detection and localization.")

    if MULTILINGUAL_AVAILABLE and multilingual_engine:
        # Simple controls row
        ctrl_col1, ctrl_col2 = st.columns([3, 1])
        with ctrl_col1:
            lang_mode_opts = {
                "auto": "🌐 Auto-Detect Language (Recommended)",
                "es": "🇪🇸 Spanish (Español)",
                "fr": "🇫🇷 French (Français)",
                "de": "🇩🇪 German (Deutsch)",
                "hi": "🇮🇳 Hindi (हिन्दी)",
                "en": "🇬🇧 English",
            }
            selected_mode = st.selectbox(
                "Language Mode:",
                list(lang_mode_opts.keys()),
                format_func=lambda k: lang_mode_opts[k],
                key="ml_chat_mode"
            )
        with ctrl_col2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat", key="clear_ml_chat", use_container_width=True):
                st.session_state.multilingual_chat_history = []
                st.rerun()

        # Query Input
        user_ml_query = st.text_input(
            "Type your question in any language:",
            key="ml_chat_input",
            placeholder="e.g., ¿Cuáles son los proyectos que haré en el curso de Python?"
        )

        if st.button("🚀 Ask Assistant", key="ml_submit_btn", use_container_width=True):
            if user_ml_query.strip():
                force_lang = None if selected_mode == "auto" else selected_mode
                with st.spinner("Processing multilingual query..."):
                    qa_chain = get_qa_chain() if LANGCHAIN_AVAILABLE else None
                    answer_data = multilingual_engine.answer_multilingual_query(
                        query=user_ml_query.strip(),
                        qa_chain=qa_chain,
                        force_language=force_lang
                    )
                    st.session_state.multilingual_chat_history.append(answer_data)

        # Display Chat History
        if st.session_state.multilingual_chat_history:
            st.divider()
            st.markdown("#### 📜 Conversation History")

            for idx, item in enumerate(reversed(st.session_state.multilingual_chat_history)):
                with st.container(border=True):
                    flag = item.get("flag", "🌐")
                    lang_name = item.get("language_name", "Unknown")
                    conf = item.get("confidence", 1.0)

                    st.markdown(f"**👤 You** ({flag} {lang_name} • {conf:.0%} match)")
                    st.markdown(f"*{item.get('original_query', '')}*")

                    st.divider()

                    st.markdown(f"**🤖 Assistant** ({flag} Localized)")
                    render_answer_box(item.get("localized_answer", ""))

                    # Pipeline Inspector — compact
                    with st.expander("🔍 Pipeline Details"):
                        st.caption(f"**Input:** {flag} {lang_name} (Confidence: {conf:.0%})")
                        st.caption(f"**English Query:** {item.get('english_query', 'Same as input')}")
                        raw_ans = item.get("raw_english_answer", "")
                        st.caption(f"**KB Answer (English):** {raw_ans[:160]}...")
                        sources = item.get("sources", [])
                        if sources:
                            for s_idx, src in enumerate(sources[:2]):
                                st.caption(f"• Source {s_idx+1}: {src[:120]}...")

        # Developer Studio — Hidden by default
        st.divider()
        with st.expander("🛠️ Developer Tools: Translation & Detection"):
            tab_trans, tab_detect = st.tabs(["🔄 Translator", "🗣️ Language Detector"])

            with tab_trans:
                c1, c2 = st.columns(2)
                with c1:
                    tr_text = st.text_area("Text to Translate:", value="Welcome to Nullclass virtual internship program!", key="dev_tr_text", height=100)
                    tr_src = st.selectbox("Source Language:", ["en", "es", "fr", "de", "hi"], index=0, key="dev_tr_src")
                with c2:
                    tr_target = st.selectbox("Target Language:", ["es", "fr", "de", "hi", "en"], index=0, key="dev_tr_tgt")
                    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                    if st.button("🚀 Translate", key="dev_tr_btn", use_container_width=True):
                        with st.spinner("Translating..."):
                            if tr_src == "en":
                                res = multilingual_engine.translate_from_english(tr_text, tr_target)
                            elif tr_target == "en":
                                res = multilingual_engine.translate_to_english(tr_text, tr_src)
                            else:
                                en_step = multilingual_engine.translate_to_english(tr_text, tr_src)
                                res = multilingual_engine.translate_from_english(en_step, tr_target)
                            with st.container(border=True):
                                st.caption(f"Translation ({tr_target.upper()}):")
                                st.markdown(f"#### {res}")

            with tab_detect:
                det_input = st.text_area("Enter any sentence:", placeholder="e.g., नमस्ते दोस्तों! / Bonjour tout le monde!", key="dev_det_input", height=90)
                if det_input and st.button("🔍 Analyze", key="dev_det_btn", use_container_width=True):
                    det_res = multilingual_engine.detect_language(det_input)
                    with st.container(border=True):
                        d_c1, d_c2, d_c3 = st.columns(3)
                        render_metric_card(det_res.get("flag", "🌐"), "Flag", d_c1)
                        render_metric_card(det_res.get("language_name", "English"), "Language", d_c2)
                        render_metric_card(f"{det_res.get('confidence', 0):.0%}", "Confidence", d_c3)
    else:
        st.info("Multilingual Engine not loaded. Ensure `multilingual_engine.py` is present.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("🤖 **AI Chatbot Platform** — Powered by Google Gemini • LangChain • FAISS • Streamlit")
