"""
langchain_helper.py - Unified LangChain helper for multi-domain Q&A chatbot.

Supports:
- Customer service FAQ retrieval (original)
- Multi-source vector store creation and retrieval
- Google Gemini / Google Palm LLM integration with fallbacks
- Dynamic knowledge base support via KnowledgeBaseManager
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_src_dir)
load_dotenv(os.path.join(_src_dir, ".env"))
load_dotenv(os.path.join(_project_dir, ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Initialization (Google Gemini → Google Palm → Fallback)
# ---------------------------------------------------------------------------

llm = None
instructor_embeddings = None

def _init_llm():
    """Initialize the LLM with fast, active Gemini models and fallback chain."""
    global llm
    if llm is not None:
        return llm

    api_key = os.environ.get("GOOGLE_API_KEY", "")

    # Candidate Gemini models in order of quota availability and speed
    gemini_candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-flash-latest",
    ]

    for model_name in gemini_candidates:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            test_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1,
                max_retries=1,
                convert_system_message_to_human=True,
            )
            # Test with a lightweight ping to verify quota
            test_llm.invoke("ping")
            llm = test_llm
            logger.info(f"✅ Initialized Google Gemini LLM ({model_name})")
            return llm
        except Exception as e:
            logger.warning(f"Model {model_name} failed ({e}); trying next candidate...")

    # Try legacy Google Palm
    try:
        from langchain_community.llms import GooglePalm
        llm = GooglePalm(google_api_key=api_key, temperature=0.1)
        logger.info("✅ Initialized Google Palm LLM")
        return llm
    except Exception as e:
        logger.warning(f"Google Palm failed: {e}")

    # Fallback: HuggingFace pipeline (offline, no API key needed)
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("✅ Initialized HuggingFace Flan-T5-small as fallback LLM")
        return llm
    except Exception as e:
        logger.warning(f"HuggingFace fallback failed: {e}")

    logger.warning("⚠️ No remote LLM initialized; SafeRetrievalQA will extract directly from vector documents.")
    return None


def _init_embeddings():
    """Initialize embeddings with fallback chain."""
    global instructor_embeddings
    if instructor_embeddings is not None:
        return instructor_embeddings

    # Try HuggingFace Instructor Embeddings
    try:
        from langchain_community.embeddings import HuggingFaceInstructEmbeddings
        instructor_embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large"
        )
        logger.info("✅ Initialized HuggingFace Instructor Embeddings")
        return instructor_embeddings
    except Exception as e:
        logger.warning(f"Instructor embeddings failed: {e}")

    # Fallback: sentence-transformers
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        instructor_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        logger.info("✅ Initialized HuggingFace sentence-transformer embeddings (fallback)")
        return instructor_embeddings
    except Exception as e:
        logger.warning(f"Sentence-transformer fallback failed: {e}")

    # Fallback: Google Generative AI Embeddings
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        instructor_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        )
        logger.info("✅ Initialized Google Generative AI Embeddings (fallback)")
        return instructor_embeddings
    except Exception as e:
        logger.warning(f"Google embeddings fallback failed: {e}")

    logger.error("❌ No embedding model could be initialized.")
    return None


# ---------------------------------------------------------------------------
# FAISS Vector DB Paths
# ---------------------------------------------------------------------------
VECTORDB_FILE_PATH = "faiss_index"


# ---------------------------------------------------------------------------
# Original Customer Service FAQ Functions (preserved & enhanced)
# ---------------------------------------------------------------------------

def create_vector_db(csv_path: str = "dataset/dataset.csv", vectordb_path: str = None):
    """Create FAISS vector database from the FAQ CSV dataset."""
    if vectordb_path is None:
        vectordb_path = VECTORDB_FILE_PATH

    embeddings = _init_embeddings()
    if embeddings is None:
        logger.error("Cannot create vector DB: no embeddings available.")
        return False

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.document_loaders.csv_loader import CSVLoader

        loader = CSVLoader(file_path=csv_path, source_column="prompt")
        data = loader.load()

        vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
        vectordb.save_local(vectordb_path)
        logger.info(f"✅ Vector DB created at '{vectordb_path}' with {len(data)} documents.")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create vector DB: {e}")
        return False


class SafeRetrievalQA:
    """Bulletproof QA chain that works across all LangChain versions.

    1. Retrieves top matching documents from FAISS vector store.
    2. If LLM is available, synthesizes an answer using the prompt.
    3. If LLM fails or is unavailable, directly extracts the verified 'response:' section from the document.
    """
    def __init__(self, retriever, llm=None, prompt_template: Optional[str] = None):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or (
            "Given the following context and a question, generate an answer based on this context only.\n"
            "In the answer try to provide as much text as possible from 'response' section in the source document context without making much changes.\n"
            "If the answer is not found in the context, kindly state 'I don't know.' Don't try to make up an answer.\n\n"
            "CONTEXT: {context}\n\n"
            "QUESTION: {question}\n\n"
            "ANSWER:"
        )

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query", "")
        if not query:
            return {"result": "Please ask a question.", "source_documents": []}

        # 1. Retrieve matching documents from vector store
        docs = []
        try:
            if hasattr(self.retriever, "invoke"):
                docs = self.retriever.invoke(query)
            elif hasattr(self.retriever, "get_relevant_documents"):
                docs = self.retriever.get_relevant_documents(query)
        except Exception as e:
            logger.warning(f"Retriever call failed: {e}")

        if not docs:
            return {"result": "I don't know.", "source_documents": []}

        # 2. Try LLM synthesis
        if self.llm is not None:
            try:
                context_str = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
                formatted_prompt = self.prompt_template.format(context=context_str, question=query)
                response = self.llm.invoke(formatted_prompt)
                ans = getattr(response, "content", str(response)).strip()
                if ans and "I don't know" not in ans:
                    return {"result": ans, "source_documents": docs}
            except Exception as ex:
                logger.warning(f"LLM generation failed ({ex}); extracting directly from documents.")

        # 3. Deterministic Document Extraction Fallback
        for doc in docs:
            content = getattr(doc, "page_content", "")
            if "response:" in content:
                parts = content.split("response:", 1)
                if len(parts) > 1 and parts[1].strip():
                    return {"result": parts[1].strip(), "source_documents": docs}

        return {
            "result": docs[0].page_content.strip(),
            "source_documents": docs
        }


def get_qa_chain(vectordb_path: str = None, custom_prompt: str = None):
    """Get a robust, version-independent RetrievalQA chain for the customer service FAQ bot."""
    if vectordb_path is None:
        vectordb_path = VECTORDB_FILE_PATH

    current_llm = _init_llm()
    embeddings = _init_embeddings()

    if embeddings is None:
        logger.error("Cannot create QA chain: embeddings unavailable.")
        return None

    try:
        from langchain_community.vectorstores import FAISS

        vectordb = FAISS.load_local(
            vectordb_path, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        return SafeRetrievalQA(retriever=retriever, llm=current_llm, prompt_template=custom_prompt)
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS index: {e}")
        return None


# ---------------------------------------------------------------------------
# Unified Multi-Domain Retriever
# ---------------------------------------------------------------------------

def create_domain_vector_db(
    data_path: str,
    vectordb_path: str,
    loader_type: str = "csv",
    source_column: str = "prompt",
    text_field: str = "text",
) -> bool:
    """Create a FAISS vector DB for any domain (medical, arxiv, etc.)."""
    embeddings = _init_embeddings()
    if embeddings is None:
        return False

    try:
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document

        documents = []

        if loader_type == "csv":
            from langchain_community.document_loaders.csv_loader import CSVLoader
            loader = CSVLoader(file_path=data_path, source_column=source_column)
            documents = loader.load()

        elif loader_type == "json":
            import json
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                text = item.get(text_field, "")
                if not text and "title" in item and "abstract" in item:
                    text = f"{item['title']}\n{item['abstract']}"
                    if "key_findings" in item:
                        text += f"\n{item['key_findings']}"
                metadata = {k: v for k, v in item.items() if k != text_field}
                documents.append(Document(page_content=text, metadata=metadata))

        if not documents:
            logger.warning("No documents loaded.")
            return False

        vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
        vectordb.save_local(vectordb_path)
        logger.info(f"✅ Domain vector DB created at '{vectordb_path}' with {len(documents)} docs.")
        return True
    except Exception as e:
        logger.error(f"❌ Domain vector DB creation failed: {e}")
        return False


def get_domain_qa_chain(vectordb_path: str, prompt_template: str) -> Any:
    """Get a QA chain for a specific domain with a custom prompt."""
    return get_qa_chain(vectordb_path=vectordb_path, custom_prompt=prompt_template)


# ---------------------------------------------------------------------------
# Utility: Get LLM response directly (for modules that need raw generation)
# ---------------------------------------------------------------------------

def get_llm_response(prompt: str) -> str:
    """Get a direct text response from the LLM."""
    current_llm = _init_llm()
    if current_llm is None:
        return "LLM is not available. Please check your API key configuration."
    try:
        response = current_llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        logger.error(f"LLM response error: {e}")
        return f"Error generating response: {str(e)}"


# ---------------------------------------------------------------------------
# Main (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== LangChain Helper Test ===")
    print(f"LLM: {_init_llm()}")
    print(f"Embeddings: {_init_embeddings()}")

    # Test vector DB creation with original dataset
    success = create_vector_db()
    if success:
        chain = get_qa_chain()
        if chain:
            result = chain({"query": "Do you provide internships?"})
            print(f"Answer: {result['result']}")
