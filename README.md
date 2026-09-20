
# 🤖 AI Chatbot Platform — Advanced Internship Extension

**Built on:** Customer Service Chatbot LLM (Nullclass Q&A System)  
**Original Stack:** Google Palm + LangChain + FAISS + Streamlit  
**Extended With:** Google Gemini, Multi-Modal AI, Medical NER, Domain Expertise, Sentiment Analysis, Multilingual Support

---

## 🎯 Project Overview

This project extends the original customer service chatbot training project with **6 advanced internship feature modules**, transforming a simple FAQ bot into a comprehensive AI-powered platform.

### Original Training Project
- Q&A system for Nullclass e-learning company using Google Palm LLM
- FAISS vector database for FAQ retrieval
- Streamlit-based user interface

### Internship Extensions (6 Modules)

| # | Module | Description |
|---|--------|-------------|
| 1 | 🏠 **Dynamic Knowledge Base** | Auto-expanding vector DB with periodic updates from CSV, JSON, URLs |
| 2 | 🎨 **Multi-Modal Hub** | Image analysis, Visual Q&A, OCR, image comparison via Gemini Vision |
| 3 | 🩺 **Medical Q&A** | MedQuAD-based clinical Q&A with medical entity recognition |
| 4 | 🔬 **arXiv Expert** | CS/AI paper search, multi-level summarization, concept visualization |
| 5 | 📊 **Sentiment Analytics** | Real-time emotion detection with adaptive response recommendations |
| 6 | 🌐 **Multilingual Support** | Auto-detection for 5 languages (EN, ES, FR, DE, HI) with translation |

---

## 📁 Project Structure

```
customer_service_chatbot_LLM/
├── README.md                          # This file
├── requirements.txt                   # All dependencies
├── dataset/
│   ├── dataset.csv                    # Original Nullclass FAQ dataset
│   ├── medquad_dataset.csv            # Medical Q&A dataset (55 entries)
│   └── arxiv_dataset.json             # CS/AI research papers (40 entries)
├── src/
│   ├── main.py                        # Multi-tab Streamlit dashboard
│   ├── langchain_helper.py            # Unified LLM & vector DB helper
│   ├── knowledge_manager.py           # Dynamic Knowledge Base Manager
│   ├── multimodal_engine.py           # Multi-Modal Engine (Gemini Vision)
│   ├── medical_qa_engine.py           # Medical Q&A with NER
│   ├── arxiv_expert_engine.py         # arXiv Domain Expert Engine
│   ├── sentiment_engine.py            # Sentiment Analysis Engine
│   └── multilingual_engine.py         # Multilingual Support Engine
```

---

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/aslin72/customer_service_chatbot_LLM.git
cd customer_service_chatbot_LLM
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API key** — Create a `.env` file:
```bash
GOOGLE_API_KEY="your_google_api_key_here"
```

---

## ▶️ Usage

```bash
cd src
streamlit run main.py
```

The app opens a **6-tab dashboard**:

### Tab 1: 🏠 Customer Service & Dynamic KB
- Ask FAQ questions about Nullclass courses
- Upload new CSV/text/URL sources to expand the knowledge base
- Check for periodic updates automatically

### Tab 2: 🎨 Multi-Modal Hub
- Upload images for AI-powered analysis
- Visual Q&A — ask questions about uploaded images
- OCR text extraction from images
- Compare two images side-by-side

### Tab 3: 🩺 Medical Q&A
- Ask medical questions (symptoms, diseases, treatments)
- See extracted medical entities highlighted
- Confidence scoring on each answer
- Browse questions by medical category
- ⚠️ Medical disclaimer included

### Tab 4: 🔬 arXiv Domain Expert
- Search CS/AI research papers
- Multi-level paper summaries (Executive / Technical / Layperson)
- Explain complex technical concepts
- Interactive concept visualization graph
- Field statistics and trending topics

### Tab 5: 📊 Sentiment Analytics
- Analyze sentiment of any text input
- See emotion classification (positive, negative, frustrated, satisfied, urgent)
- Get adaptive response recommendations
- Session-wide analytics dashboard with charts

### Tab 6: 🌐 Multilingual
- Automatic language detection
- Translate between English, Spanish, French, German, Hindi
- Test language detection with sample text
- Configure language preferences

---

## 🔧 Module Details

### Dynamic Knowledge Base (`knowledge_manager.py`)
- **Incremental updates** using `FAISS.merge_from()`
- **Change detection** via SHA256 file hashing
- **Version tracking** with metadata persistence
- **Multi-source ingestion**: CSV, JSON, raw text, URLs
- **Periodic update checks** for automated freshness

### Multi-Modal Engine (`multimodal_engine.py`)
- **Google Gemini Vision** integration (gemini-2.0-flash)
- **Graceful degradation** — works without API key (limited)
- Image analysis, Visual Q&A, OCR, metadata generation, comparison

### Medical Q&A (`medical_qa_engine.py`)
- **Dictionary-based Medical NER** — symptoms, diseases, treatments, body parts
- **FAISS retrieval** with keyword fallback
- **Confidence scoring** based on entity overlap and answer quality
- **Safety disclaimers** on every response

### arXiv Expert (`arxiv_expert_engine.py`)
- **Inverted keyword index** for fast offline search
- **Optional FAISS vector search** for semantic matching
- **Multi-level summarization**: Executive, Technical, Layperson
- **Concept graph** data generation for interactive visualization
- **Follow-up Q&A** with conversation history

### Sentiment Engine (`sentiment_engine.py`)
- **Lexicon-based scoring** — no external ML models needed
- **Negation and intensifier handling** for accuracy
- **6 emotion categories**: positive, negative, neutral, frustrated, satisfied, urgent
- **Adaptive response modifiers** — tone, temperature, prefix/suffix suggestions
- **Session analytics** with trend analysis

### Multilingual Engine (`multilingual_engine.py`)
- **Multi-tier language detection**: Unicode script → langdetect → indicator words
- **5 languages**: English, Spanish, French, German, Hindi
- **Translation via Gemini** with pre-translated phrase fallback
- **Culturally appropriate** greetings and response formatting

---

## 📊 Datasets

| Dataset | Source | Size | Format |
|---------|--------|------|--------|
| `dataset.csv` | Nullclass FAQ | ~200 entries | CSV (prompt, response) |
| `medquad_dataset.csv` | MedQuAD-inspired | 55 entries | CSV (question, answer, category, focus) |
| `arxiv_dataset.json` | arXiv CS/AI | 40 papers | JSON (id, title, abstract, authors, categories, year, key_findings) |

---

## 🛠️ Technology Stack

- **LLMs**: Google Gemini, Google Palm, HuggingFace Flan-T5 (fallback)
- **Embeddings**: HuggingFace Instructor, sentence-transformers, Google Embeddings
- **Vector Store**: FAISS
- **Framework**: LangChain
- **UI**: Streamlit
- **Visualization**: Plotly
- **NLP**: NLTK, spaCy, TextBlob, VADER

---

## 📝 Sample Questions

**Customer Service:**
- Do you guys provide internship and also do you offer EMI payments?
- Do you have a JavaScript course?

**Medical Q&A:**
- What are the symptoms of diabetes?
- How is hypertension treated?

**arXiv Expert:**
- Explain the attention mechanism in transformers
- What are the latest advances in federated learning?

**Sentiment Test:**
- "I'm really frustrated with the service quality. This is unacceptable!"
- "Thank you so much! The course was excellent and very helpful!"

**Multilingual:**
- "Hola, ¿cómo puedo registrarme para un curso?"
- "Bonjour, quels cours proposez-vous?"