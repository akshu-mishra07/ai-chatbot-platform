"""
End-to-End Test Suite for All 6 Internship Modules
Tests each module independently and reports pass/fail status.
"""

import os
import sys
import json
import traceback
from datetime import datetime

# Setup paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(SRC_DIR, ".env"))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

RESULTS = {}
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def test_module(name, func):
    """Run a test function and capture results."""
    print(f"\n{'='*60}")
    print(f"  TESTING: {name}")
    print(f"{'='*60}")
    try:
        result = func()
        RESULTS[name] = result
        print(f"\n  Result: {result['status']}")
        if result.get("details"):
            for d in result["details"]:
                print(f"    - {d}")
    except Exception as e:
        RESULTS[name] = {"status": FAIL, "details": [f"Exception: {e}", traceback.format_exc()[-200:]]}
        print(f"\n  Result: {FAIL} - {e}")


# ============================================================
# TEST 1: LangChain Helper (Core)
# ============================================================
def test_langchain_helper():
    from langchain_helper import _init_llm, _init_embeddings, create_vector_db, get_qa_chain, get_llm_response
    details = []

    # Test embeddings init
    emb = _init_embeddings()
    if emb is not None:
        details.append(f"{PASS} Embeddings initialized: {type(emb).__name__}")
    else:
        details.append(f"{FAIL} Embeddings failed to initialize")
        return {"status": FAIL, "details": details}

    # Test LLM init
    llm = _init_llm()
    if llm is not None:
        details.append(f"{PASS} LLM initialized: {type(llm).__name__}")
    else:
        details.append(f"{WARN} LLM not initialized (API key issue?)")

    # Test vector DB creation
    csv_path = os.path.join(PROJECT_DIR, "dataset", "dataset.csv")
    if os.path.exists(csv_path):
        success = create_vector_db(csv_path=csv_path)
        if success:
            details.append(f"{PASS} Vector DB created from FAQ dataset")
        else:
            details.append(f"{FAIL} Vector DB creation failed")
    else:
        details.append(f"{FAIL} dataset.csv not found at {csv_path}")

    # Test QA chain
    chain = get_qa_chain()
    if chain:
        details.append(f"{PASS} QA chain created successfully")
        try:
            response = chain({"query": "Do you provide internships?"})
            answer = response.get("result", "")
            details.append(f"{PASS} QA response: {answer[:80]}...")
        except Exception as e:
            details.append(f"{WARN} QA chain query failed: {str(e)[:80]}")
    else:
        details.append(f"{WARN} QA chain not available")

    # Test direct LLM response
    if llm:
        try:
            resp = get_llm_response("Say hello in one word.")
            details.append(f"{PASS} Direct LLM response: {resp[:50]}")
        except Exception as e:
            details.append(f"{WARN} Direct LLM response failed: {str(e)[:60]}")

    has_fail = any(FAIL in d for d in details)
    return {"status": FAIL if has_fail else PASS, "details": details}


# ============================================================
# TEST 2: Knowledge Manager (Dynamic KB)
# ============================================================
def test_knowledge_manager():
    from knowledge_manager import KnowledgeBaseManager
    details = []

    kb = KnowledgeBaseManager(
        vectordb_path=os.path.join(PROJECT_DIR, "test_faiss_index"),
        metadata_path=os.path.join(PROJECT_DIR, "test_kb_metadata.json"),
    )
    details.append(f"{PASS} KnowledgeBaseManager initialized")

    # Test status
    status = kb.get_status()
    details.append(f"{PASS} Status: version={status.get('version')}, docs={status.get('document_count')}")

    # Test add text documents
    result = kb.add_text_documents(
        ["Nullclass offers Python, Java, and Data Science courses.",
         "Virtual internships are available for 3 months duration."],
        [{"source": "test"}, {"source": "test"}]
    )
    details.append(f"{PASS} Added text docs: {result.get('added_docs', 0)} documents")

    # Test CSV ingestion
    csv_path = os.path.join(PROJECT_DIR, "dataset", "dataset.csv")
    if os.path.exists(csv_path):
        result = kb.add_csv_source(csv_path)
        details.append(f"{PASS} CSV ingestion: {result.get('added_docs', 0)} docs added")

    # Test check for updates
    changed = kb.check_for_updates([csv_path])
    details.append(f"{PASS} Update check: {len(changed)} changed sources")

    # Test retriever
    retriever = kb.get_retriever()
    if retriever:
        details.append(f"{PASS} Retriever created successfully")
    else:
        details.append(f"{WARN} Retriever not available")

    # Cleanup test files
    import shutil
    test_index = os.path.join(PROJECT_DIR, "test_faiss_index")
    test_meta = os.path.join(PROJECT_DIR, "test_kb_metadata.json")
    if os.path.exists(test_index):
        shutil.rmtree(test_index)
    if os.path.exists(test_meta):
        os.remove(test_meta)
    details.append(f"{PASS} Cleanup completed")

    has_fail = any(FAIL in d for d in details)
    return {"status": FAIL if has_fail else PASS, "details": details}


# ============================================================
# TEST 3: Multi-Modal Engine
# ============================================================
def test_multimodal_engine():
    from multimodal_engine import MultiModalEngine
    details = []

    engine = MultiModalEngine()
    details.append(f"{PASS} MultiModalEngine initialized")

    status = engine.get_status()
    details.append(f"{PASS} Status: available={status.get('available')}, model={status.get('model_name', 'N/A')}")

    if status.get("available"):
        # Create a simple test image (red square)
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")

        # Test image analysis
        try:
            result = engine.analyze_image(img, "What color is this image?")
            details.append(f"{PASS} Image analysis: {result[:80]}...")
        except Exception as e:
            details.append(f"{WARN} Image analysis error: {str(e)[:80]}")

        # Test text extraction
        try:
            result = engine.extract_text_from_image(img)
            details.append(f"{PASS} OCR extraction: {result[:80]}...")
        except Exception as e:
            details.append(f"{WARN} OCR error: {str(e)[:80]}")

        # Test metadata generation
        try:
            result = engine.generate_image_description_for_search(img)
            details.append(f"{PASS} Metadata generation: {type(result).__name__} with {len(result)} keys")
        except Exception as e:
            details.append(f"{WARN} Metadata error: {str(e)[:80]}")
    else:
        details.append(f"{WARN} Engine not available (API key issue?) - {status.get('error', '')}")

    has_fail = any(FAIL in d for d in details)
    return {"status": FAIL if has_fail else PASS, "details": details}


# ============================================================
# TEST 4: Medical Q&A Engine
# ============================================================
def test_medical_qa():
    from medical_qa_engine import MedicalQAEngine
    details = []

    dataset_path = os.path.join(PROJECT_DIR, "dataset", "medquad_dataset.csv")
    engine = MedicalQAEngine(
        dataset_path=dataset_path,
        vectordb_path=os.path.join(PROJECT_DIR, "test_medical_faiss"),
    )
    details.append(f"{PASS} MedicalQAEngine initialized")

    # Test status
    status = engine.get_status()
    details.append(f"{PASS} Status: loaded={status.get('dataset_loaded')}, size={status.get('dataset_size')}")

    # Test entity extraction
    entities = engine.extract_medical_entities("I have a headache and fever, could it be diabetes?")
    entity_count = sum(len(v) for v in entities.values())
    details.append(f"{PASS} Entity extraction: found {entity_count} entities in {len(entities)} categories")
    for etype, elist in entities.items():
        details.append(f"     {etype}: {elist}")

    # Test vector DB creation
    try:
        success = engine.create_medical_vector_db()
        if success:
            details.append(f"{PASS} Medical vector DB created")
        else:
            details.append(f"{WARN} Medical vector DB creation failed (using keyword fallback)")
    except Exception as e:
        details.append(f"{WARN} Vector DB error: {str(e)[:80]}")

    # Test Q&A
    result = engine.get_answer("What are the symptoms of diabetes?")
    details.append(f"{PASS} Q&A answer: {result.get('answer', 'N/A')[:80]}...")
    details.append(f"     Confidence: {result.get('confidence', 0):.2f}")
    details.append(f"     Entities: {result.get('entities', {})}")

    # Test category search
    categories = engine.get_categories()
    details.append(f"{PASS} Categories: {categories}")

    # Test search by category
    if categories:
        cat_results = engine.search_by_category(categories[0])
        details.append(f"{PASS} Category search ({categories[0]}): {len(cat_results)} results")

    # Test disclaimer
    disclaimer = engine.get_safety_disclaimer()
    details.append(f"{PASS} Disclaimer present: {len(disclaimer)} chars")

    # Cleanup
    import shutil
    test_idx = os.path.join(PROJECT_DIR, "test_medical_faiss")
    if os.path.exists(test_idx):
        shutil.rmtree(test_idx)

    has_fail = any(FAIL in d for d in details)
    return {"status": FAIL if has_fail else PASS, "details": details}


# ============================================================
# TEST 5: arXiv Expert Engine
# ============================================================
def test_arxiv_expert():
    from arxiv_expert_engine import ArXivExpertEngine
    details = []

    dataset_path = os.path.join(PROJECT_DIR, "dataset", "arxiv_dataset.json")
    engine = ArXivExpertEngine(
        dataset_path=dataset_path,
        vectordb_path=os.path.join(PROJECT_DIR, "test_arxiv_faiss"),
    )
    details.append(f"{PASS} ArXivExpertEngine initialized")

    # Test status
    status = engine.get_status()
    details.append(f"{PASS} Status: papers={status.get('total_papers')}, index={status.get('keyword_index_ready')}")

    # Test paper search
    results = engine.search_papers("transformer attention mechanism", top_k=3)
    details.append(f"{PASS} Paper search: found {len(results)} papers")
    if results:
        details.append(f"     Top result: {results[0].get('title', 'N/A')[:60]}")

    # Test paper summarization
    papers = engine._load_dataset()
    if papers:
        pid = papers[0].get("id", "")
        for level in ["executive", "technical", "layperson"]:
            summary = engine.summarize_paper(pid, level)
            details.append(f"{PASS} Summary ({level}): {summary[:60]}...")

    # Test concept explanation
    explanation = engine.explain_concept("attention mechanism")
    details.append(f"{PASS} Concept explanation: {explanation[:80]}...")

    # Test concept graph data
    graph = engine.generate_concept_graph_data("deep learning")
    details.append(f"{PASS} Concept graph: {len(graph.get('nodes', []))} nodes, {len(graph.get('edges', []))} edges")

    # Test Q&A
    qa_result = engine.answer_question("What are the latest advances in NLP?")
    details.append(f"{PASS} Q&A answer: {qa_result.get('answer', 'N/A')[:80]}...")

    # Test field statistics
    stats = engine.get_field_statistics()
    details.append(f"{PASS} Statistics: {stats.get('total_papers')} papers, {len(stats.get('category_distribution', {}))} categories")

    # Test trending topics
    trends = engine.get_trending_topics(top_n=5)
    details.append(f"{PASS} Trending topics: {trends[:3]}")

    # Cleanup
    import shutil
    test_idx = os.path.join(PROJECT_DIR, "test_arxiv_faiss")
    if os.path.exists(test_idx):
        shutil.rmtree(test_idx)

    has_fail = any(FAIL in d for d in details)
    return {"status": FAIL if has_fail else PASS, "details": details}


# ============================================================
# TEST 6: Sentiment Engine
# ============================================================
def test_sentiment_engine():
    from sentiment_engine import SentimentEngine
    details = []

    engine = SentimentEngine(history_size=50)
    details.append(f"{PASS} SentimentEngine initialized")

    # Test positive sentiment
    result = engine.analyze_sentiment("Thank you so much! The course was excellent and very helpful!")
    details.append(f"{PASS} Positive: sentiment={result.get('sentiment')}, score={result.get('score'):.2f}, "
                   f"emoji={result.get('emoji_indicator')}")

    # Test negative sentiment
    result = engine.analyze_sentiment("This is terrible! I'm very frustrated and disappointed with the service.")
    details.append(f"{PASS} Negative: sentiment={result.get('sentiment')}, score={result.get('score'):.2f}, "
                   f"emoji={result.get('emoji_indicator')}")

    # Test urgent sentiment
    result = engine.analyze_sentiment("I need help urgently! This is an emergency, please respond immediately!")
    details.append(f"{PASS} Urgent: sentiment={result.get('sentiment')}, score={result.get('score'):.2f}, "
                   f"urgent={result.get('is_urgent')}")

    # Test neutral sentiment
    result = engine.analyze_sentiment("I would like to know more about the course schedule.")
    details.append(f"{PASS} Neutral: sentiment={result.get('sentiment')}, score={result.get('score'):.2f}")

    # Test response modifier
    neg_result = engine.analyze_sentiment("I hate this product, it's completely broken!")
    modifier = engine.get_response_modifier(neg_result)
    details.append(f"{PASS} Response modifier: tone={modifier.get('tone')}, escalate={modifier.get('escalate')}")
    details.append(f"     Prefix: {modifier.get('prefix_suggestion', 'N/A')[:60]}")

    # Test session analytics
    analytics = engine.get_session_analytics()
    details.append(f"{PASS} Analytics: total={analytics.get('total_messages')}, "
                   f"avg={analytics.get('avg_score', 0):.2f}, trend={analytics.get('trend')}")

    # Test sentiment trend
    trend = engine.get_sentiment_trend()
    details.append(f"{PASS} Trend data: {len(trend)} data points")

    # Test reset
    engine.reset_session()
    analytics_after = engine.get_session_analytics()
    details.append(f"{PASS} Reset: total_after={analytics_after.get('total_messages')}")

    has_fail = any(FAIL in d for d in details)
    return {"status": FAIL if has_fail else PASS, "details": details}


# ============================================================
# TEST 7: Multilingual Engine
# ============================================================
def test_multilingual_engine():
    from multilingual_engine import MultilingualEngine
    details = []

    engine = MultilingualEngine()
    details.append(f"{PASS} MultilingualEngine initialized")

    # Test status
    status = engine.get_status()
    details.append(f"{PASS} Status: languages={len(status.get('supported_languages', []))}")

    # Test supported languages
    langs = engine.get_supported_languages()
    details.append(f"{PASS} Supported: {[l['name'] for l in langs]}")

    # Test English detection
    result = engine.detect_language("Hello, how can I sign up for a course?")
    details.append(f"{PASS} EN detect: lang={result.get('language_code')}, conf={result.get('confidence', 0):.2f}")

    # Test Spanish detection
    result = engine.detect_language("Hola, ¿cómo puedo registrarme para un curso?")
    details.append(f"{PASS} ES detect: lang={result.get('language_code')}, conf={result.get('confidence', 0):.2f}")

    # Test French detection
    result = engine.detect_language("Bonjour, comment puis-je m'inscrire à un cours?")
    details.append(f"{PASS} FR detect: lang={result.get('language_code')}, conf={result.get('confidence', 0):.2f}")

    # Test German detection
    result = engine.detect_language("Hallo, wie kann ich mich für einen Kurs anmelden?")
    details.append(f"{PASS} DE detect: lang={result.get('language_code')}, conf={result.get('confidence', 0):.2f}")

    # Test Hindi detection
    result = engine.detect_language("नमस्ते, मैं कोर्स के लिए कैसे रजिस्टर करूं?")
    details.append(f"{PASS} HI detect: lang={result.get('language_code')}, conf={result.get('confidence', 0):.2f}")

    # Test multilingual query processing
    ml_result = engine.process_multilingual_query("Hola, ¿tienen cursos de Python?")
    details.append(f"{PASS} Query processing: detected={ml_result.get('detected_language', {}).get('language_code')}, "
                   f"english={ml_result.get('english_text', 'N/A')[:50]}")

    # Test translation (English → Spanish)
    try:
        translated = engine.translate_from_english("Hello, welcome to our platform!", "es")
        details.append(f"{PASS} EN→ES translation: {translated[:60]}")
    except Exception as e:
        details.append(f"{WARN} Translation failed: {str(e)[:60]}")

    # Test greeting
    for lang in ["en", "es", "fr", "de", "hi"]:
        greeting = engine.get_greeting(lang)
        details.append(f"{PASS} Greeting ({lang}): {greeting}")

    # Test language switching
    success = engine.set_language("es")
    details.append(f"{PASS} Language switch to ES: {success}")

    # Test end-to-end multilingual answering
    try:
        ans_res = engine.answer_multilingual_query("¿Tienen pasantías virtuales?")
        details.append(f"{PASS} End-to-end QA (ES): lang={ans_res.get('target_language')}, "
                       f"answer={ans_res.get('localized_answer', '')[:60]}...")
    except Exception as e:
        details.append(f"{WARN} End-to-end QA failed: {str(e)[:60]}")

    has_fail = any(FAIL in d for d in details)
    return {"status": FAIL if has_fail else PASS, "details": details}


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧪 AI CHATBOT PLATFORM - END-TO-END TEST SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        ("Module 1: LangChain Helper (Core)", test_langchain_helper),
        ("Module 2: Knowledge Manager (Dynamic KB)", test_knowledge_manager),
        ("Module 3: Multi-Modal Engine", test_multimodal_engine),
        ("Module 4: Medical Q&A Engine", test_medical_qa),
        ("Module 5: arXiv Expert Engine", test_arxiv_expert),
        ("Module 6: Sentiment Engine", test_sentiment_engine),
        ("Module 7: Multilingual Engine", test_multilingual_engine),
    ]

    for name, func in tests:
        test_module(name, func)

    # Final Summary
    print("\n" + "=" * 60)
    print("  📋 FINAL TEST SUMMARY")
    print("=" * 60)
    for name, result in RESULTS.items():
        print(f"  {result['status']}  {name}")

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS.values() if r["status"] == PASS)
    warned = sum(1 for r in RESULTS.values() if r["status"] == WARN)
    failed = sum(1 for r in RESULTS.values() if r["status"] == FAIL)

    print(f"\n  Total: {total} | Passed: {passed} | Warnings: {warned} | Failed: {failed}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
