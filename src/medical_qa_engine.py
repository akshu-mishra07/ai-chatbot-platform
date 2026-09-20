"""
Medical Q&A Engine with FAISS Vector Retrieval and Medical Entity Recognition (NER).
Provides fast medical question-answering, entity extraction, confidence scoring,
and safety disclaimers for healthcare-oriented chatbots.
"""

import os
import re
import csv
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Try importing LangChain & FAISS components with graceful fallbacks
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        FAISS = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None
        HuggingFaceInstructEmbeddings = None

try:
    from langchain_community.document_loaders import CSVLoader
except ImportError:
    try:
        from langchain.document_loaders.csv_loader import CSVLoader
    except ImportError:
        CSVLoader = None

try:
    from langchain.docstore.document import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        Document = None

_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_src_dir)
load_dotenv(os.path.join(_src_dir, ".env"))
load_dotenv(os.path.join(_project_dir, ".env"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Medical entity dictionaries for NER
MEDICAL_ENTITIES: Dict[str, List[str]] = {
    "symptoms": [
        "headache", "fever", "cough", "fatigue", "nausea", "dizziness",
        "chest pain", "shortness of breath", "back pain", "joint pain",
        "sore throat", "runny nose", "vomiting", "diarrhea", "insomnia",
        "anxiety", "depression", "rash", "swelling", "numbness",
        "muscle pain", "abdominal pain", "weight loss", "weight gain",
        "blurred vision", "palpitations", "constipation", "bloating"
    ],
    "diseases": [
        "diabetes", "hypertension", "asthma", "arthritis", "cancer",
        "heart disease", "stroke", "alzheimer", "parkinson", "epilepsy",
        "pneumonia", "bronchitis", "tuberculosis", "hepatitis", "covid",
        "influenza", "malaria", "anemia", "osteoporosis", "fibromyalgia",
        "migraine", "copd", "thyroid", "kidney disease", "liver disease"
    ],
    "treatments": [
        "ibuprofen", "acetaminophen", "aspirin", "antibiotic", "insulin",
        "chemotherapy", "radiation", "surgery", "physical therapy", "vaccine",
        "antidepressant", "blood pressure medication", "inhaler", "steroid",
        "painkiller", "anti-inflammatory", "immunotherapy", "dialysis",
        "transfusion", "transplant"
    ],
    "body_parts": [
        "heart", "lung", "liver", "kidney", "brain", "stomach",
        "intestine", "bone", "muscle", "skin", "eye", "ear",
        "throat", "spine", "joint", "blood", "nerve"
    ]
}


class MedicalQAEngine:
    """Medical Q&A engine with FAISS retrieval and medical entity recognition."""

    def __init__(
        self,
        dataset_path: str = "dataset/medquad_dataset.csv",
        vectordb_path: str = "medical_faiss_index"
    ) -> None:
        """Initialize the Medical Q&A Engine.

        Args:
            dataset_path: Path to the medical Q&A dataset CSV.
            vectordb_path: Path to store / load the FAISS vector database.
        """
        self.dataset_path = self._resolve_path(dataset_path)
        self.vectordb_path = str(self._resolve_path(vectordb_path))
        self.vector_store_ready = False
        self.vector_store = None
        self.embeddings = None

        # Initialize embeddings model
        self._init_embeddings()

        # Load dataset into memory
        self.dataset: List[Dict[str, str]] = self._load_dataset()

        # Attempt to load existing FAISS index if available
        self._try_load_vector_store()

    def _resolve_path(self, relative_or_abs_path: str) -> Path:
        """Resolve a path against cwd and project root directory."""
        path = Path(relative_or_abs_path)
        if path.is_absolute() and path.exists():
            return path
        
        # Check relative to cwd
        if path.exists():
            return path.resolve()
        
        # Check relative to script directory / parent
        script_dir = Path(__file__).resolve().parent
        parent_dir = script_dir.parent
        if (parent_dir / relative_or_abs_path).exists():
            return (parent_dir / relative_or_abs_path).resolve()
        if (script_dir / relative_or_abs_path).exists():
            return (script_dir / relative_or_abs_path).resolve()

        # Fallback to parent dir combined path
        return parent_dir / relative_or_abs_path

    def _init_embeddings(self) -> None:
        """Try to initialize HuggingFace embeddings model with fallbacks."""
        # 1. Try HuggingFaceInstructEmbeddings
        if HuggingFaceInstructEmbeddings is not None:
            try:
                logger.info("Initializing HuggingFaceInstructEmbeddings (hkunlp/instructor-large)...")
                self.embeddings = HuggingFaceInstructEmbeddings(
                    model_name="hkunlp/instructor-large"
                )
                logger.info("HuggingFaceInstructEmbeddings successfully initialized.")
                return
            except Exception as e:
                logger.warning(f"Could not load HuggingFaceInstructEmbeddings: {e}. Trying fallback...")

        # 2. Try standard HuggingFaceEmbeddings (all-MiniLM-L6-v2)
        if HuggingFaceEmbeddings is not None:
            try:
                logger.info("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("HuggingFaceEmbeddings successfully initialized.")
                return
            except Exception as e:
                logger.warning(f"Could not load HuggingFaceEmbeddings: {e}.")

        logger.warning("Embeddings could not be initialized. FAISS vector search will be disabled; using fallback search.")

    def _try_load_vector_store(self) -> bool:
        """Attempt to load an existing FAISS vector store from disk."""
        if FAISS is None or self.embeddings is None:
            return False
        
        index_dir = Path(self.vectordb_path)
        if index_dir.exists() and (index_dir / "index.faiss").exists():
            try:
                logger.info(f"Loading FAISS index from {self.vectordb_path}...")
                self.vector_store = FAISS.load_local(
                    self.vectordb_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.vector_store_ready = True
                logger.info("FAISS vector store loaded successfully.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing FAISS index: {e}")
                self.vector_store_ready = False
        return False

    def _load_dataset(self) -> List[Dict[str, str]]:
        """Load CSV into list of dicts with keys: question, answer, category, focus.

        Returns:
            List of dictionary items containing Q&A records, or empty list on error.
        """
        records: List[Dict[str, str]] = []
        target_path = Path(self.dataset_path)

        if not target_path.exists():
            # Try alternate file in dataset/
            alt_path = target_path.parent / "dataset.csv"
            if alt_path.exists():
                target_path = alt_path
            else:
                logger.warning(f"Dataset file not found at {self.dataset_path}.")
                return records

        try:
            with open(target_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Normalize keys
                    row_clean = {k.strip().lower() if k else "": (v.strip() if v else "") for k, v in row.items()}
                    
                    question = (
                        row_clean.get("question")
                        or row_clean.get("prompt")
                        or row_clean.get("query")
                        or ""
                    )
                    answer = (
                        row_clean.get("answer")
                        or row_clean.get("response")
                        or ""
                    )
                    category = (
                        row_clean.get("category")
                        or row_clean.get("type")
                        or "General Medical"
                    )
                    focus = (
                        row_clean.get("focus")
                        or row_clean.get("topic")
                        or row_clean.get("disease")
                        or ""
                    )

                    if question and answer:
                        records.append({
                            "question": question,
                            "answer": answer,
                            "category": category,
                            "focus": focus
                        })
            logger.info(f"Loaded {len(records)} records from {target_path}.")
        except Exception as e:
            logger.error(f"Error loading dataset from {target_path}: {e}")
            return []

        return records

    def create_medical_vector_db(self) -> bool:
        """Create FAISS vector store from the medical dataset and save it locally.

        Returns:
            bool: True if vector store was created and saved successfully, False otherwise.
        """
        if FAISS is None:
            logger.error("FAISS library is not installed or available.")
            return False

        if self.embeddings is None:
            self._init_embeddings()
            if self.embeddings is None:
                logger.error("Cannot create vector database without an embedding model.")
                return False

        if not self.dataset:
            self.dataset = self._load_dataset()
            if not self.dataset:
                logger.error("Cannot create vector database: No data loaded.")
                return False

        try:
            docs: List[Any] = []
            
            # If CSVLoader and Document are available, use Document creation
            if Document is not None:
                for item in self.dataset:
                    content = f"Question: {item['question']}\nAnswer: {item['answer']}"
                    metadata = {
                        "question": item["question"],
                        "answer": item["answer"],
                        "category": item["category"],
                        "focus": item["focus"]
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
                
                logger.info(f"Building FAISS vector store from {len(docs)} documents...")
                self.vector_store = FAISS.from_documents(documents=docs, embedding=self.embeddings)
            else:
                texts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in self.dataset]
                metadatas = [
                    {
                        "question": item["question"],
                        "answer": item["answer"],
                        "category": item["category"],
                        "focus": item["focus"]
                    }
                    for item in self.dataset
                ]
                logger.info(f"Building FAISS vector store from {len(texts)} texts...")
                self.vector_store = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)

            # Ensure parent directory for vectordb exists
            Path(self.vectordb_path).parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(self.vectordb_path)
            self.vector_store_ready = True
            logger.info(f"FAISS vector store saved successfully at {self.vectordb_path}.")
            return True

        except Exception as e:
            logger.error(f"Failed to create medical vector database: {e}")
            self.vector_store_ready = False
            return False

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text using dictionary-based NER.

        Args:
            text: Input string (e.g. user question or answer text).

        Returns:
            Dictionary of matched entity categories and lists of found entities.
            Example: {"symptoms": ["headache", "fever"], "diseases": ["diabetes"]}
        """
        if not text:
            return {}

        text_lower = text.lower()
        extracted: Dict[str, List[str]] = {}

        for category, entity_list in MEDICAL_ENTITIES.items():
            matched_entities: List[str] = []
            for entity in entity_list:
                entity_lower = entity.lower()
                # Use regex with word boundaries for accurate entity extraction
                pattern = r'\b' + re.escape(entity_lower) + r'\b'
                if re.search(pattern, text_lower):
                    if entity_lower not in matched_entities:
                        matched_entities.append(entity_lower)
            
            if matched_entities:
                extracted[category] = matched_entities

        return extracted

    def get_confidence_score(self, query: str, answer: str) -> float:
        """Calculate a confidence score based on entity overlap and answer length.

        Args:
            query: The user query string.
            answer: The candidate answer string.

        Returns:
            Float confidence score between 0.0 and 1.0.
        """
        if not query or not answer:
            return 0.0

        query_entities = self.extract_medical_entities(query)
        answer_entities = self.extract_medical_entities(answer)

        # Flatten entity sets
        q_set = {item for sublist in query_entities.values() for item in sublist}
        a_set = {item for sublist in answer_entities.values() for item in sublist}

        # Calculate entity overlap
        if q_set:
            overlap_count = len(q_set.intersection(a_set))
            entity_score = overlap_count / len(q_set)
        else:
            # Fallback to token matching if no predefined entities found in query
            stop_words = {"what", "is", "the", "are", "how", "to", "for", "a", "an", "of", "and", "in", "do", "can", "should"}
            q_tokens = {w for w in re.findall(r'\b\w+\b', query.lower()) if w not in stop_words and len(w) > 2}
            a_tokens = {w for w in re.findall(r'\b\w+\b', answer.lower()) if w not in stop_words and len(w) > 2}
            if q_tokens:
                overlap_count = len(q_tokens.intersection(a_tokens))
                entity_score = min(overlap_count / max(len(q_tokens), 1), 1.0)
            else:
                entity_score = 0.5

        # Factor in answer length: longer answers provide more comprehensive information (up to 0.2 bonus)
        word_count = len(answer.split())
        length_bonus = min(word_count / 100.0, 1.0) * 0.2

        # Weighted combination: 70% entity overlap + 20% length factor + 10% baseline presence
        confidence = (0.70 * entity_score) + length_bonus + 0.10
        confidence = max(0.0, min(confidence, 1.0))

        return round(confidence, 2)

    def get_safety_disclaimer(self) -> str:
        """Return standard medical disclaimer text."""
        return (
            "DISCLAIMER: The information provided is for educational and informational "
            "purposes only and does not constitute medical advice, diagnosis, or treatment. "
            "Always consult a qualified healthcare provider for personal health concerns. "
            "If you are experiencing a medical emergency, please call your local emergency services immediately."
        )

    def _keyword_search_fallback(self, question: str) -> Tuple[str, str, List[str]]:
        """Fallback search matching keywords and entities against in-memory dataset."""
        if not self.dataset:
            return (
                "I am sorry, but the medical knowledge base is currently empty or unavailable.",
                "General Medical",
                []
            )

        q_lower = question.lower()
        extracted_q = self.extract_medical_entities(question)
        q_entities = {e for cat in extracted_q.values() for e in cat}
        words = set(re.findall(r'\b\w+\b', q_lower)) - {"what", "is", "the", "are", "how", "to", "for", "a", "an", "of", "and", "in"}

        best_match = None
        best_score = -1.0

        for item in self.dataset:
            item_q = item["question"].lower()
            item_focus = item.get("focus", "").lower()
            
            score = 0.0
            # Entity match bonus
            for ent in q_entities:
                if ent in item_focus or ent in item_q:
                    score += 3.0
            
            # Word match bonus
            for word in words:
                if len(word) > 2 and (word in item_q or word in item_focus):
                    score += 1.0

            if score > best_score:
                best_score = score
                best_match = item

        if best_match and best_score > 0:
            return (
                best_match["answer"],
                best_match.get("category", "General Medical"),
                [best_match["question"]]
            )

        return (
            "I could not find specific medical information matching your question in the knowledge base. "
            "Please consult a doctor or healthcare specialist for detailed advice.",
            "General Medical",
            []
        )

    def get_answer(self, question: str) -> Dict[str, Any]:
        """Get an answer for a medical question with entity extraction and confidence scoring.

        Args:
            question: The user query string.

        Returns:
            Dict containing answer, entities, confidence, category, disclaimer, sources.
        """
        if not question or not question.strip():
            return {
                "answer": "Please ask a valid medical question.",
                "entities": {},
                "confidence": 0.0,
                "category": "General Medical",
                "disclaimer": self.get_safety_disclaimer(),
                "sources": []
            }

        entities = self.extract_medical_entities(question)
        answer_text = ""
        category = "General Medical"
        sources: List[str] = []

        # 1. Try vector store retrieval if available
        if self.vector_store_ready and self.vector_store is not None:
            try:
                docs = self.vector_store.similarity_search(question, k=2)
                if docs:
                    top_doc = docs[0]
                    # Extract answer and metadata
                    metadata = getattr(top_doc, "metadata", {})
                    answer_text = metadata.get("answer")
                    category = metadata.get("category", "General Medical")
                    
                    if not answer_text:
                        # Parse page_content if metadata not populated
                        content = getattr(top_doc, "page_content", "")
                        if "Answer:" in content:
                            answer_text = content.split("Answer:", 1)[1].strip()
                        else:
                            answer_text = content

                    sources = [doc.metadata.get("question", doc.page_content[:60]) for doc in docs if hasattr(doc, "metadata")]
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keyword search: {e}")
                answer_text = ""

        # 2. Fallback to keyword matching if vector store is unavailable or returned empty
        if not answer_text:
            answer_text, category, sources = self._keyword_search_fallback(question)

        # 3. Calculate confidence score
        confidence = self.get_confidence_score(question, answer_text)

        # 4. Return structured response
        return {
            "answer": answer_text,
            "entities": entities,
            "confidence": confidence,
            "category": category,
            "disclaimer": self.get_safety_disclaimer(),
            "sources": sources
        }

    def search_by_category(self, category: str) -> List[Dict[str, str]]:
        """Filter dataset by category, return matching Q&A pairs.

        Args:
            category: Category name to filter by.

        Returns:
            List of matching Q&A dictionary records.
        """
        if not category:
            return []

        cat_clean = category.strip().lower()
        results: List[Dict[str, str]] = []

        for row in self.dataset:
            row_cat = row.get("category", "").strip().lower()
            if cat_clean in row_cat or row_cat in cat_clean:
                results.append(row)

        return results

    def search_by_entity(self, entity: str) -> List[Dict[str, str]]:
        """Search dataset for rows where 'focus' or 'question' matches the entity.

        Args:
            entity: Entity or term to search for.

        Returns:
            List of matching Q&A dictionary records.
        """
        if not entity:
            return []

        ent_clean = entity.strip().lower()
        results: List[Dict[str, str]] = []

        for row in self.dataset:
            focus = row.get("focus", "").strip().lower()
            question = row.get("question", "").strip().lower()
            if ent_clean in focus or ent_clean in question:
                results.append(row)

        return results

    def get_categories(self) -> List[str]:
        """Return unique sorted list of categories from dataset."""
        categories = set()
        for row in self.dataset:
            cat = row.get("category", "").strip()
            if cat:
                categories.add(cat)
        return sorted(list(categories))

    def get_status(self) -> Dict[str, Any]:
        """Return status dictionary of the medical QA engine.

        Returns:
            Dict containing dataset_loaded, dataset_size, vector_store_ready, categories.
        """
        return {
            "dataset_loaded": len(self.dataset) > 0,
            "dataset_size": len(self.dataset),
            "vector_store_ready": self.vector_store_ready,
            "categories": self.get_categories()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Medical Q&A Engine Initialization & Demonstration")
    print("=" * 60)

    # Initialize the engine
    engine = MedicalQAEngine(
        dataset_path="dataset/medquad_dataset.csv",
        vectordb_path="medical_faiss_index"
    )

    # Display engine status
    status = engine.get_status()
    print(f"\n[Status]: {json.dumps(status, indent=2)}")

    # Sample Entity Extraction Demo
    sample_text = "Patient reports acute headache, high fever, and chest pain, with a history of diabetes and hypertension taking insulin."
    extracted = engine.extract_medical_entities(sample_text)
    print(f"\n[NER Extraction on Sample Text]:")
    print(f"Text: '{sample_text}'")
    print(f"Extracted Entities: {json.dumps(extracted, indent=2)}")

    # Sample Medical Q&A Query
    test_query = "What are the common symptoms of diabetes and how is insulin used?"
    print(f"\n[Q&A Query]: '{test_query}'")
    response = engine.get_answer(test_query)
    print(f"\n[Response]:")
    print(f"Answer: {response['answer']}")
    print(f"Entities: {response['entities']}")
    print(f"Confidence: {response['confidence']}")
    print(f"Category: {response['category']}")
    print(f"Sources: {response['sources']}")
    print(f"Disclaimer: {response['disclaimer']}")

    # Category and Entity Search Demos
    categories = engine.get_categories()
    print(f"\n[Available Categories]: {categories}")
    if categories:
        first_cat = categories[0]
        cat_matches = engine.search_by_category(first_cat)
        print(f"Found {len(cat_matches)} entries in category '{first_cat}'")

    entity_matches = engine.search_by_entity("diabetes")
    print(f"Found {len(entity_matches)} entries matching entity 'diabetes'")
