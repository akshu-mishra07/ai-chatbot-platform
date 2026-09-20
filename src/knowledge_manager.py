"""
Dynamic Knowledge Base Expansion Module for Customer Service Chatbot.

This module manages the FAISS vector database lifecycle, including incremental
document ingestion (CSV, JSON, URLs, raw text), file hash-based change detection,
metadata versioning, periodic updates, and retriever generation.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import urllib.request
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports for LangChain components across different versions
try:
    from langchain.vectorstores import FAISS
except ImportError:
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        FAISS = None
        logger.warning("FAISS vectorstore module could not be imported.")

try:
    from langchain.embeddings import HuggingFaceInstructEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceInstructEmbeddings
    except ImportError:
        HuggingFaceInstructEmbeddings = None
        logger.warning("HuggingFaceInstructEmbeddings module could not be imported.")

try:
    from langchain.document_loaders.csv_loader import CSVLoader
except ImportError:
    try:
        from langchain_community.document_loaders import CSVLoader
    except ImportError:
        CSVLoader = None

try:
    from langchain.schema import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        try:
            from langchain_core.documents import Document
        except ImportError:
            class Document:  # Fallback minimal Document class
                def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None


class KnowledgeBaseManager:
    """Manages dynamic expansion, versioning, and periodic updates of the FAISS vector knowledge base."""

    def __init__(
        self,
        vectordb_path: str = "faiss_index",
        metadata_path: str = "kb_metadata.json",
        embedding_model_name: str = "hkunlp/instructor-large",
    ) -> None:
        """Initialize the KnowledgeBaseManager.

        Args:
            vectordb_path: Path to the FAISS vector database folder.
            metadata_path: Path to the JSON metadata file.
            embedding_model_name: Name of the HuggingFace instructor embedding model.
        """
        self.vectordb_path = Path(vectordb_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.vectordb = None

        # Initialize Embeddings
        self._init_embeddings()

        # Load or create metadata
        self.metadata = self._load_metadata()

        # Attempt to load existing vector database if present
        self._load_existing_vectordb()

    def _init_embeddings(self) -> None:
        """Initialize the Hugging Face Instructor embeddings model."""
        if HuggingFaceInstructEmbeddings is None:
            logger.warning("HuggingFaceInstructEmbeddings is unavailable. Embeddings operations will be skipped or mocked.")
            return

        try:
            logger.info(f"Initializing embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceInstructEmbeddings(model_name=self.embedding_model_name)
            logger.info("Embeddings successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceInstructEmbeddings ({self.embedding_model_name}): {e}")
            self.embeddings = None

    def _load_existing_vectordb(self) -> bool:
        """Load an existing FAISS vector store from disk if available.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if self.embeddings is None or FAISS is None:
            return False

        if not self.vectordb_path.exists():
            return False

        index_file = self.vectordb_path / "index.faiss"
        pkl_file = self.vectordb_path / "index.pkl"
        if not (index_file.exists() or pkl_file.exists() or self.vectordb_path.is_dir()):
            return False

        try:
            logger.info(f"Loading existing FAISS vector store from {self.vectordb_path}")
            try:
                # Newer LangChain requires allow_dangerous_deserialization=True for pickle
                self.vectordb = FAISS.load_local(
                    str(self.vectordb_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except TypeError:
                self.vectordb = FAISS.load_local(
                    str(self.vectordb_path),
                    self.embeddings,
                )
            logger.info("Existing FAISS vector store loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load existing vector store from {self.vectordb_path}: {e}")
            self.vectordb = None
            return False

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata JSON from disk or return default structure.

        Returns:
            Dict[str, Any]: Loaded or default metadata dictionary.
        """
        default_metadata: Dict[str, Any] = {
            "version": 0,
            "document_count": 0,
            "sources": {},
            "last_updated": None,
            "document_hashes": {},
        }

        if not self.metadata_path.exists():
            return default_metadata

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Ensure all required keys exist
                    for key, val in default_metadata.items():
                        if key not in data:
                            data[key] = val
                    return data
                return default_metadata
        except Exception as e:
            logger.error(f"Error reading metadata from {self.metadata_path}: {e}. Initializing defaults.")
            return default_metadata

    def _save_metadata(self) -> bool:
        """Persist metadata to disk.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata to {self.metadata_path}: {e}")
            return False

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file content for change detection.

        Args:
            file_path: Path to the target file.

        Returns:
            str: Hexadecimal SHA256 hash string, or empty string on error.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.warning(f"File not found for hash computation: {file_path}")
            return ""

        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def _merge_and_persist(
        self,
        docs: List[Document],
        source_id: str,
        source_type: str,
        source_hash: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Helper to create or merge documents into the FAISS vector database and update metadata.

        Args:
            docs: List of Document objects to index.
            source_id: Identifier or file path of the source.
            source_type: Type of source ('csv', 'json', 'url', 'text').
            source_hash: Optional SHA256 hash of the source.
            extra_metadata: Additional metadata dictionary.

        Returns:
            Dict[str, Any]: Operation status report.
        """
        if not docs:
            return {
                "status": "error",
                "message": "No documents provided to add.",
                "source": source_id,
                "added_docs": 0,
            }

        if self.embeddings is None or FAISS is None:
            return {
                "status": "error",
                "message": "FAISS or Embedding model is not initialized.",
                "source": source_id,
                "added_docs": 0,
            }

        try:
            # If no in-memory vector store, check disk again or create new
            if self.vectordb is None:
                self._load_existing_vectordb()

            if self.vectordb is None:
                logger.info(f"Creating new FAISS vector database from {len(docs)} documents.")
                self.vectordb = FAISS.from_documents(documents=docs, embedding=self.embeddings)
            else:
                logger.info(f"Merging {len(docs)} documents into existing FAISS vector database.")
                new_db = FAISS.from_documents(documents=docs, embedding=self.embeddings)
                self.vectordb.merge_from(new_db)

            # Persist vector database to disk
            self.vectordb_path.mkdir(parents=True, exist_ok=True)
            self.vectordb.save_local(str(self.vectordb_path))

            # Update tracking metadata
            timestamp = datetime.now().isoformat()
            new_version = self.metadata.get("version", 0) + 1
            new_doc_count = self.metadata.get("document_count", 0) + len(docs)

            self.metadata["version"] = new_version
            self.metadata["document_count"] = new_doc_count
            self.metadata["last_updated"] = timestamp

            source_info: Dict[str, Any] = {
                "type": source_type,
                "added_at": timestamp,
                "documents_count": len(docs),
                "hash": source_hash,
            }
            if extra_metadata:
                source_info.update(extra_metadata)

            self.metadata["sources"][source_id] = source_info
            if source_hash:
                self.metadata["document_hashes"][source_id] = source_hash

            self._save_metadata()

            return {
                "status": "success",
                "message": f"Successfully ingested {len(docs)} documents from {source_id}.",
                "source": source_id,
                "source_type": source_type,
                "added_docs": len(docs),
                "new_version": new_version,
                "total_documents": new_doc_count,
                "last_updated": timestamp,
            }

        except Exception as e:
            logger.error(f"Error indexing documents from {source_id}: {e}")
            return {
                "status": "error",
                "message": f"Failed to ingest documents: {str(e)}",
                "source": source_id,
                "added_docs": 0,
            }

    def add_csv_source(self, csv_path: str, source_column: str = "prompt") -> Dict[str, Any]:
        """Load CSV, check if file hash changed since last ingest, and update the vector store.

        Args:
            csv_path: Path to the CSV file.
            source_column: Column name to use as document source/content identifier.

        Returns:
            Dict[str, Any]: Status dictionary containing ingestion results.
        """
        try:
            path = Path(csv_path)
            if not path.exists():
                return {
                    "status": "error",
                    "message": f"CSV file not found: {csv_path}",
                    "source": csv_path,
                    "added_docs": 0,
                }

            current_hash = self._compute_file_hash(csv_path)
            stored_hash = self.metadata.get("document_hashes", {}).get(csv_path)

            if stored_hash and current_hash == stored_hash:
                logger.info(f"CSV source {csv_path} has not changed (hash: {current_hash[:8]}...). Skipping.")
                return {
                    "status": "skipped",
                    "message": "File has not changed since last ingestion.",
                    "source": csv_path,
                    "added_docs": 0,
                    "version": self.metadata.get("version", 0),
                    "total_documents": self.metadata.get("document_count", 0),
                }

            logger.info(f"Loading CSV data from {csv_path} (source column: {source_column})...")
            docs: List[Document] = []

            # Try using LangChain's CSVLoader first
            if CSVLoader is not None:
                try:
                    loader = CSVLoader(file_path=str(path), source_column=source_column, encoding="utf-8")
                    docs = loader.load()
                except Exception as load_err:
                    logger.warning(f"CSVLoader failed with default encoding: {load_err}. Falling back to manual CSV parsing.")
                    docs = []

            # Fallback manual CSV loader if CSVLoader is unavailable or fails
            if not docs:
                import csv
                for enc in ["utf-8-sig", "utf-8", "latin-1"]:
                    try:
                        with open(path, mode="r", encoding=enc) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                content = row.get(source_column) or "\n".join([f"{k}: {v}" for k, v in row.items() if v])
                                if content:
                                    meta = {"source": csv_path, **{k: v for k, v in row.items() if k != source_column}}
                                    docs.append(Document(page_content=str(content), metadata=meta))
                        if docs:
                            break
                    except Exception as parse_err:
                        logger.debug(f"Failed to read CSV with encoding {enc}: {parse_err}")

            if not docs:
                return {
                    "status": "error",
                    "message": f"No valid documents could be parsed from CSV: {csv_path}",
                    "source": csv_path,
                    "added_docs": 0,
                }

            return self._merge_and_persist(
                docs=docs,
                source_id=csv_path,
                source_type="csv",
                source_hash=current_hash,
                extra_metadata={"source_column": source_column},
            )

        except Exception as e:
            logger.error(f"Error in add_csv_source for {csv_path}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "source": csv_path,
                "added_docs": 0,
            }

    def add_text_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Accept raw text strings, create Document objects, embed and merge into vector store.

        Args:
            texts: List of text strings to index.
            metadatas: Optional list of metadata dictionaries corresponding to each text.

        Returns:
            Dict[str, Any]: Status dictionary with result of addition.
        """
        try:
            if not texts:
                return {
                    "status": "error",
                    "message": "Empty text list provided.",
                    "added_docs": 0,
                }

            docs: List[Document] = []
            timestamp = datetime.now().isoformat()

            for i, text in enumerate(texts):
                if not text or not str(text).strip():
                    continue
                meta = metadatas[i] if (metadatas and i < len(metadatas) and isinstance(metadatas[i], dict)) else {}
                if "source" not in meta:
                    meta["source"] = "raw_text"
                if "timestamp" not in meta:
                    meta["timestamp"] = timestamp
                docs.append(Document(page_content=str(text).strip(), metadata=meta))

            if not docs:
                return {
                    "status": "error",
                    "message": "No valid non-empty text strings found.",
                    "added_docs": 0,
                }

            batch_id = f"text_batch_{int(datetime.now().timestamp())}"
            return self._merge_and_persist(
                docs=docs,
                source_id=batch_id,
                source_type="text",
                source_hash=None,
            )

        except Exception as e:
            logger.error(f"Error in add_text_documents: {e}")
            return {
                "status": "error",
                "message": str(e),
                "added_docs": 0,
            }

    def add_json_source(self, json_path: str, text_field: str = "text") -> Dict[str, Any]:
        """Load JSON (list of dicts or nested structure), extract text_field, create docs and merge.

        Args:
            json_path: Path to the JSON file.
            text_field: Key name containing the main text content.

        Returns:
            Dict[str, Any]: Status dictionary.
        """
        try:
            path = Path(json_path)
            if not path.exists():
                return {
                    "status": "error",
                    "message": f"JSON file not found: {json_path}",
                    "source": json_path,
                    "added_docs": 0,
                }

            current_hash = self._compute_file_hash(json_path)
            stored_hash = self.metadata.get("document_hashes", {}).get(json_path)

            if stored_hash and current_hash == stored_hash:
                logger.info(f"JSON source {json_path} has not changed. Skipping.")
                return {
                    "status": "skipped",
                    "message": "File has not changed since last ingestion.",
                    "source": json_path,
                    "added_docs": 0,
                    "version": self.metadata.get("version", 0),
                    "total_documents": self.metadata.get("document_count", 0),
                }

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs: List[Document] = []
            items = data if isinstance(data, list) else [data]

            for idx, item in enumerate(items):
                if isinstance(item, dict):
                    content = item.get(text_field)
                    if not content:
                        # Concatenate string values if text_field is not found
                        content = "\n".join([f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float))])
                    
                    if content:
                        meta = {"source": json_path, "item_index": idx, **{k: v for k, v in item.items() if k != text_field and isinstance(v, (str, int, float, bool))}}
                        docs.append(Document(page_content=str(content), metadata=meta))
                elif isinstance(item, str) and item.strip():
                    docs.append(Document(page_content=item.strip(), metadata={"source": json_path, "item_index": idx}))

            if not docs:
                return {
                    "status": "error",
                    "message": f"No valid documents found in JSON file {json_path} using text_field='{text_field}'.",
                    "source": json_path,
                    "added_docs": 0,
                }

            return self._merge_and_persist(
                docs=docs,
                source_id=json_path,
                source_type="json",
                source_hash=current_hash,
                extra_metadata={"text_field": text_field},
            )

        except Exception as e:
            logger.error(f"Error in add_json_source for {json_path}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "source": json_path,
                "added_docs": 0,
            }

    def add_url_source(self, url: str) -> Dict[str, Any]:
        """Fetch text content from a URL, chunk into ~500 char segments with overlap, and add to vector store.

        Args:
            url: Web page URL to fetch and index.

        Returns:
            Dict[str, Any]: Status dictionary.
        """
        try:
            logger.info(f"Fetching content from URL: {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KnowledgeBaseManager/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                raw_bytes = response.read()
                # Attempt decoding
                charset = response.headers.get_content_charset() or "utf-8"
                html_text = raw_bytes.decode(charset, errors="replace")

            # Extract clean text from HTML
            clean_text = ""
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_text, "html.parser")
                for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    element.decompose()
                clean_text = soup.get_text(separator=" ", strip=True)
            except Exception:
                # Fallback simple regex stripping
                clean_text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
                clean_text = re.sub(r"<[^>]+>", " ", clean_text)
                clean_text = " ".join(clean_text.split())

            if not clean_text or len(clean_text.strip()) < 10:
                return {
                    "status": "error",
                    "message": f"Insufficient readable text extracted from URL: {url}",
                    "source": url,
                    "added_docs": 0,
                }

            # Chunk the text into ~500 character segments with overlap
            chunks: List[str] = []
            if RecursiveCharacterTextSplitter is not None:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks = splitter.split_text(clean_text)
            else:
                # Manual chunking fallback
                chunk_size = 500
                chunk_overlap = 50
                step = chunk_size - chunk_overlap
                for start in range(0, len(clean_text), step):
                    chunk = clean_text[start : start + chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)

            timestamp = datetime.now().isoformat()
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"source": url, "chunk_index": idx, "total_chunks": len(chunks), "timestamp": timestamp},
                )
                for idx, chunk in enumerate(chunks)
            ]

            url_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

            return self._merge_and_persist(
                docs=docs,
                source_id=url,
                source_type="url",
                source_hash=url_hash,
                extra_metadata={"chunks_count": len(chunks)},
            )

        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "source": url,
                "added_docs": 0,
            }

    def get_status(self) -> Dict[str, Any]:
        """Return current metadata status: version, doc count, sources, and last updated.

        Returns:
            Dict[str, Any]: Status summary dictionary.
        """
        vectordb_exists = False
        if self.vectordb_path.exists():
            faiss_file = self.vectordb_path / "index.faiss"
            pkl_file = self.vectordb_path / "index.pkl"
            vectordb_exists = faiss_file.exists() or pkl_file.exists() or bool(list(self.vectordb_path.glob("*")))

        return {
            "version": self.metadata.get("version", 0),
            "document_count": self.metadata.get("document_count", 0),
            "sources": self.metadata.get("sources", {}),
            "last_updated": self.metadata.get("last_updated"),
            "vectordb_path": str(self.vectordb_path),
            "vectordb_exists": vectordb_exists,
            "embedding_model": self.embedding_model_name,
            "embeddings_ready": self.embeddings is not None,
            "vectordb_loaded": self.vectordb is not None,
        }

    def check_for_updates(self, source_paths: List[str]) -> List[str]:
        """Compare current file hashes with stored hashes and return paths that have changed.

        Args:
            source_paths: List of file paths to inspect.

        Returns:
            List[str]: Paths of modified or new files.
        """
        changed: List[str] = []
        stored_hashes = self.metadata.get("document_hashes", {})

        for path_str in source_paths:
            path = Path(path_str)
            if not path.exists() or not path.is_file():
                continue

            current_hash = self._compute_file_hash(path_str)
            stored_hash = stored_hashes.get(path_str)

            if stored_hash is None or current_hash != stored_hash:
                changed.append(path_str)

        return changed

    def periodic_update(
        self,
        source_paths: List[str],
        source_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Check all source paths for changes, re-ingest changed sources, and return a summary.

        Args:
            source_paths: List of source paths or URLs to check.
            source_types: Optional mapping of path -> type ('csv', 'json', 'url', 'text').

        Returns:
            Dict[str, Any]: Summary of updated and unchanged sources.
        """
        source_types = source_types or {}
        results: Dict[str, Any] = {}
        updated_sources: List[str] = []
        skipped_sources: List[str] = []
        errors: List[str] = []

        logger.info(f"Starting periodic update check for {len(source_paths)} sources...")

        for source in source_paths:
            try:
                stype = source_types.get(source)
                if not stype:
                    if source.startswith("http://") or source.startswith("https://"):
                        stype = "url"
                    elif source.endswith(".csv"):
                        stype = "csv"
                    elif source.endswith(".json"):
                        stype = "json"
                    elif source.endswith(".txt"):
                        stype = "text"
                    else:
                        stype = "csv"

                if stype == "csv":
                    res = self.add_csv_source(source)
                elif stype == "json":
                    res = self.add_json_source(source)
                elif stype == "url":
                    res = self.add_url_source(source)
                elif stype == "text":
                    path = Path(source)
                    if path.exists():
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        res = self.add_text_documents([content], metadatas=[{"source": source}])
                    else:
                        res = {"status": "error", "message": f"Text file not found: {source}"}
                else:
                    res = {"status": "error", "message": f"Unsupported source type: {stype}"}

                results[source] = res
                if res.get("status") == "success":
                    updated_sources.append(source)
                elif res.get("status") == "skipped":
                    skipped_sources.append(source)
                else:
                    errors.append(source)

            except Exception as e:
                logger.error(f"Error during periodic update for source {source}: {e}")
                err_res = {"status": "error", "message": str(e)}
                results[source] = err_res
                errors.append(source)

        return {
            "timestamp": datetime.now().isoformat(),
            "checked_count": len(source_paths),
            "updated_count": len(updated_sources),
            "skipped_count": len(skipped_sources),
            "error_count": len(errors),
            "updated_sources": updated_sources,
            "skipped_sources": skipped_sources,
            "errors": errors,
            "results": results,
            "current_status": self.get_status(),
        }

    def get_retriever(self, score_threshold: float = 0.7) -> Any:
        """Load the FAISS vector store and return a retriever.

        Args:
            score_threshold: Minimum similarity score threshold for query retrieval.

        Returns:
            Vector store retriever object, or None if the vector store does not exist.
        """
        try:
            if self.vectordb is None:
                loaded = self._load_existing_vectordb()
                if not loaded:
                    logger.warning("No FAISS vector store exists or could be loaded. Returning None.")
                    return None

            logger.info(f"Creating retriever with score_threshold={score_threshold}")
            try:
                return self.vectordb.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": score_threshold},
                )
            except Exception:
                # Direct argument fallback as used in existing langchain_helper.py
                return self.vectordb.as_retriever(score_threshold=score_threshold)

        except Exception as e:
            logger.error(f"Failed to get retriever: {e}")
            return None


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: KnowledgeBaseManager Standalone Execution")
    print("=" * 60)

    # Initialize manager in a test environment
    demo_db_path = "test_faiss_index"
    demo_meta_path = "test_kb_metadata.json"
    
    manager = KnowledgeBaseManager(
        vectordb_path=demo_db_path,
        metadata_path=demo_meta_path,
    )

    print("\n1. Initial Knowledge Base Status:")
    print(json.dumps(manager.get_status(), indent=2))

    print("\n2. Ingesting Sample Text Documents...")
    sample_texts = [
        "Welcome to our Customer Support! Our support team is available 24/7 via live chat.",
        "To reset your account password, click on 'Forgot Password' on the login screen.",
        "We offer a 30-day money-back guarantee for all subscription plans.",
    ]
    sample_metadatas = [
        {"category": "support_hours", "source": "demo_text"},
        {"category": "auth", "source": "demo_text"},
        {"category": "billing", "source": "demo_text"},
    ]
    res_text = manager.add_text_documents(sample_texts, sample_metadatas)
    print("Result:", json.dumps(res_text, indent=2))

    print("\n3. Ingesting Sample JSON Source...")
    sample_json_path = "temp_demo_faq.json"
    sample_json_data = [
        {"text": "How do I update my billing info?", "category": "billing", "response": "Navigate to Settings -> Billing."},
        {"text": "Where is my order confirmation?", "category": "orders", "response": "Check your registered email inbox."},
    ]
    with open(sample_json_path, "w", encoding="utf-8") as f:
        json.dump(sample_json_data, f, indent=2)

    res_json = manager.add_json_source(sample_json_path, text_field="text")
    print("Result:", json.dumps(res_json, indent=2))

    print("\n4. Checking for Updates on JSON Source (Unchanged):")
    changed_files = manager.check_for_updates([sample_json_path])
    print(f"Changed files detected: {changed_files}")

    print("\n5. Re-ingesting JSON Source (Should be skipped):")
    res_skip = manager.add_json_source(sample_json_path, text_field="text")
    print("Result:", json.dumps(res_skip, indent=2))

    print("\n6. Final Knowledge Base Status:")
    status = manager.get_status()
    print(json.dumps(status, indent=2))

    print("\n7. Retrieving Document Retriever:")
    retriever = manager.get_retriever(score_threshold=0.5)
    print("Retriever instance:", retriever)

    # Cleanup demo files if needed
    try:
        if Path(sample_json_path).exists():
            Path(sample_json_path).unlink()
        if Path(demo_meta_path).exists():
            Path(demo_meta_path).unlink()
        import shutil
        if Path(demo_db_path).exists():
            shutil.rmtree(demo_db_path)
        print("\nCleanup completed.")
    except Exception as cleanup_err:
        print(f"Cleanup note: {cleanup_err}")

    print("\nDemo completed successfully.")
