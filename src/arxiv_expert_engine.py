"""
Domain Expert Engine for arXiv Research Papers in Computer Science and AI.

Provides semantic search, multi-level paper summarization, concept explanations,
domain question answering, related paper discovery, knowledge graph data generation,
and field trend analytics.
"""

import os
import re
import json
import math
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
from collections import Counter, defaultdict
from dotenv import load_dotenv

_src_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_src_dir)
load_dotenv(os.path.join(_src_dir, ".env"))
load_dotenv(os.path.join(_project_dir, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard stop words for text preprocessing and keyword indexing
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "also", "using", "used", "paper", "propose", "proposed", "show", "shows", "demonstrate", "presents",
    "present", "based", "results", "model", "models", "approach", "method", "methods", "across", "via",
    "new", "well", "one", "two", "three", "first", "second", "many", "state", "art"
}

CATEGORY_MAP = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.DC": "Distributed Computing",
    "cs.IR": "Information Retrieval",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "stat.ML": "Machine Learning (Stats)",
}


class ArXivExpertEngine:
    """Domain expert chatbot and analysis engine for arXiv CS/AI papers."""

    def __init__(self, dataset_path: str = "dataset/arxiv_dataset.json", vectordb_path: str = "arxiv_faiss_index"):
        """Initialize the ArXiv Expert Engine with dataset, index structures, and optional vector/LLM integrations."""
        self.dataset_path = dataset_path
        self.vectordb_path = vectordb_path
        self.papers: List[Dict[str, Any]] = []
        self.paper_id_map: Dict[str, Dict[str, Any]] = {}
        self.keyword_index: Dict[str, Set[int]] = defaultdict(set)
        self.paper_tokens: List[Counter] = []
        self.vector_store_ready: bool = False
        self.vectordb = None
        self.embeddings = None
        self.llm = None
        self.embedding_model_name: Optional[str] = None

        # 1. Load dataset
        self.papers = self._load_dataset()
        self.paper_id_map = {str(p.get("id", "")).strip(): p for p in self.papers}

        # 2. Build keyword index
        self._build_keyword_index()

        # 3. Initialize embeddings and LLM
        self._init_embeddings_and_llm()

        # 4. Try loading pre-existing vector store if available
        self._try_load_vector_db()

        logger.info(
            f"ArXivExpertEngine initialized with {len(self.papers)} papers. "
            f"Vector store ready: {self.vector_store_ready}, LLM available: {self.llm is not None}"
        )

    def _resolve_path(self, target_path: str) -> Path:
        """Resolve a relative path across possible execution root directories."""
        p = Path(target_path)
        if p.is_absolute() and p.exists():
            return p
        if p.exists():
            return p

        # Check relative to script directory
        script_dir = Path(__file__).resolve().parent
        candidate1 = script_dir / target_path
        if candidate1.exists():
            return candidate1

        # Check relative to project parent directory
        candidate2 = script_dir.parent / target_path
        if candidate2.exists():
            return candidate2

        return p

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load paper records from the JSON dataset or load built-in fallback records."""
        resolved_path = self._resolve_path(self.dataset_path)

        if resolved_path.exists():
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"Loaded {len(data)} papers from {resolved_path}")
                        return data
                    elif isinstance(data, dict) and "papers" in data:
                        logger.info(f"Loaded {len(data['papers'])} papers from {resolved_path}")
                        return data["papers"]
            except Exception as e:
                logger.warning(f"Failed to load dataset from {resolved_path}: {e}")

        logger.info("Using embedded default AI/CS arXiv paper dataset.")
        return self._get_default_papers()

    def _get_default_papers(self) -> List[Dict[str, Any]]:
        """Fallback built-in dataset containing foundational and recent AI papers."""
        return [
            {
                "id": "1706.03762",
                "title": "Attention Is All You Need",
                "abstract": (
                    "The dominant sequence transduction models are based on complex recurrent or convolutional neural "
                    "networks that include an encoder and a decoder. We propose the Transformer, an architecture based "
                    "solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two "
                    "machine translation tasks show superior quality while being more parallelizable and faster to train."
                ),
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
                "categories": ["cs.CL", "cs.LG", "cs.AI"],
                "year": 2017,
                "key_findings": [
                    "Introduced the Transformer architecture based purely on multi-head self-attention mechanisms.",
                    "Eliminated recurrence and convolutions, enabling highly parallelized GPU training.",
                    "Established new state-of-the-art BLEU scores on WMT translation benchmarks."
                ]
            },
            {
                "id": "1810.04805",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "abstract": (
                    "We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed "
                    "to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on left and right context. "
                    "Pre-trained BERT can be fine-tuned with just one additional output layer to create state-of-the-art models for 11 NLP tasks."
                ),
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                "categories": ["cs.CL", "cs.AI", "cs.LG"],
                "year": 2018,
                "key_findings": [
                    "Introduced Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) for bidirectional pre-training.",
                    "Demonstrated that unified pre-trained representations eliminate heavy task-specific architecture engineering.",
                    "Advanced the state-of-the-art across 11 NLP benchmarks including GLUE, MultiNLI, and SQuAD."
                ]
            },
            {
                "id": "2005.14165",
                "title": "Language Models are Few-Shot Learners",
                "abstract": (
                    "We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance. "
                    "We train GPT-3, an autoregressive language model with 175 billion parameters, and evaluate performance on dozens of NLP datasets "
                    "under zero-shot, one-shot, and few-shot conditions without any gradient updates or fine-tuning."
                ),
                "authors": ["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah", "Jared Kaplan", "Dario Amodei"],
                "categories": ["cs.CL", "cs.AI", "cs.LG"],
                "year": 2020,
                "key_findings": [
                    "Demonstrated that 175B parameter language models exhibit strong in-context few-shot learning.",
                    "Identified predictable power-law scaling laws between parameters, dataset compute, and loss.",
                    "Showed emergent reasoning, translation, arithmetic, and code generation capabilities."
                ]
            },
            {
                "id": "1512.03385",
                "title": "Deep Residual Learning for Image Recognition",
                "abstract": (
                    "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training "
                    "of networks that are substantially deeper than those used previously. We reformulate layers as learning residual functions "
                    "with reference to layer inputs. We evaluate residual nets with depth of up to 152 layers on ImageNet."
                ),
                "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
                "categories": ["cs.CV", "cs.LG", "cs.AI"],
                "year": 2015,
                "key_findings": [
                    "Introduced skip connections (identity mappings) to overcome vanishing/exploding gradients in deep networks.",
                    "Won 1st place in ILSVRC 2015 classification with a 3.57% top-5 error rate.",
                    "Enabled successful training of networks with over 100 to 1000 layers."
                ]
            },
            {
                "id": "2106.09685",
                "title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "abstract": (
                    "Full fine-tuning of large pre-trained models is computationally expensive. We propose Low-Rank Adaptation (LoRA), "
                    "which freezes pre-trained model weights and injects trainable rank decomposition matrices into Transformer layers, "
                    "drastically reducing the number of trainable parameters for downstream tasks with zero inference latency overhead."
                ),
                "authors": ["Edward J. Hu", "Yelong Shen", "Phillip Wallis", "Zeyuan Allen-Zhu", "Weizhu Chen"],
                "categories": ["cs.CL", "cs.LG", "cs.AI"],
                "year": 2021,
                "key_findings": [
                    "Reduced trainable parameters by up to 10,000x and GPU VRAM usage by 3x compared to full fine-tuning.",
                    "Matched or outperformed full fine-tuning across GPT-3, RoBERTa, and DeBERTa.",
                    "Allowed modular multi-task deployment by switching adapter weights without changing base model."
                ]
            },
            {
                "id": "2205.14135",
                "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
                "abstract": (
                    "Transformers are slow and memory-intensive on long sequences due to quadratic attention complexity. "
                    "We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to compute exact softmax attention "
                    "with reduced GPU memory reads/writes between high bandwidth memory (HBM) and on-chip SRAM."
                ),
                "authors": ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Re"],
                "categories": ["cs.LG", "cs.AI", "cs.DC"],
                "year": 2022,
                "key_findings": [
                    "Designed an IO-aware tiling algorithm computing exact softmax attention without materializing full N x N matrix in HBM.",
                    "Achieved 2-4x speedup in training wall-clock time for Transformer models.",
                    "Extended feasible Transformer sequence lengths to 8k, 16k, and beyond."
                ]
            },
            {
                "id": "2006.11239",
                "title": "Denoising Diffusion Probabilistic Models",
                "abstract": (
                    "We present high-quality image synthesis results using diffusion probabilistic models, a class of latent variable models "
                    "inspired by non-equilibrium thermodynamics. Our results are obtained by training on a weighted variational bound designed "
                    "according to a novel connection between diffusion models and denoising score matching with Langevin dynamics."
                ),
                "authors": ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"],
                "categories": ["cs.LG", "cs.CV", "stat.ML"],
                "year": 2020,
                "key_findings": [
                    "Demonstrated diffusion models produce sample quality competitive with GANs.",
                    "Formulated equivalence between score matching and progressive Gaussian denoising.",
                    "Established the mathematical foundations of modern image generation systems like Stable Diffusion."
                ]
            },
            {
                "id": "2005.11401",
                "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                "abstract": (
                    "We explore general-purpose fine-tuning recipes for Retrieval-Augmented Generation (RAG) — models which combine pre-trained "
                    "parametric memory (sequence-to-sequence language model) and non-parametric memory (dense vector index of Wikipedia) "
                    "accessed via maximum inner product search."
                ),
                "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Douwe Kiela"],
                "categories": ["cs.CL", "cs.AI", "cs.IR"],
                "year": 2020,
                "key_findings": [
                    "Combined parametric neural generator with non-parametric dense vector retriever.",
                    "Substantially reduced factual hallucinations on open-domain question answering.",
                    "Enabled updating factual knowledge simply by swapping the indexed vector corpus."
                ]
            },
            {
                "id": "2203.02155",
                "title": "Training language models to follow instructions with human feedback",
                "abstract": (
                    "We show an avenue for aligning language models with user intent by fine-tuning with human feedback. "
                    "Using Reinforcement Learning from Human Feedback (RLHF), we train InstructGPT models that are much better at following "
                    "instructions, generating truthful outputs, and reducing toxic generations."
                ),
                "authors": ["Long Ouyang", "Jeff Wu", "Xu Jiang", "John Schulman", "Ilya Sutskever"],
                "categories": ["cs.CL", "cs.AI", "cs.LG"],
                "year": 2022,
                "key_findings": [
                    "Established the standard 3-phase alignment workflow: SFT, Reward Modeling, and PPO Policy Optimization.",
                    "Proved that a 1.3B aligned InstructGPT model is preferred by annotators over a 175B unaligned GPT-3 model.",
                    "Formed the core alignment foundation for ChatGPT and modern instruction-following LLMs."
                ]
            },
            {
                "id": "2103.00020",
                "title": "Learning Transferable Visual Models From Natural Language Supervision",
                "abstract": (
                    "We demonstrate that predicting which caption goes with which image is an efficient and scalable way to learn "
                    "SOTA visual representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. "
                    "We call this approach CLIP (Contrastive Language-Image Pre-training)."
                ),
                "authors": ["Alec Radford", "Jong Wook Kim", "Chris Hallacy", "Aditya Ramesh", "Ilya Sutskever"],
                "categories": ["cs.CV", "cs.CL", "cs.LG", "cs.AI"],
                "year": 2021,
                "key_findings": [
                    "Introduced contrastive multimodal pre-training bridging natural language descriptions and vision representations.",
                    "Achieved robust zero-shot classification matching a supervised ResNet-50 without task fine-tuning.",
                    "Provided joint embedding space utilized in modern text-to-image diffusion models."
                ]
            },
            {
                "id": "2305.18290",
                "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                "abstract": (
                    "We propose Direct Preference Optimization (DPO), an algorithm to implicitly optimize language model policy "
                    "under the Bradley-Terry preference model without training an explicit reward model or sampling from the policy during training. "
                    "DPO optimizes policy parameters directly using a closed-form binary cross-entropy loss."
                ),
                "authors": ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"],
                "categories": ["cs.LG", "cs.AI", "cs.CL"],
                "year": 2023,
                "key_findings": [
                    "Derived exact closed-form mapping between Bradley-Terry reward models and optimal language model policies.",
                    "Eliminated the need for complex reinforcement learning loops and reward model training.",
                    "Achieved equal or superior alignment stability and performance compared to RLHF on summarization."
                ]
            },
            {
                "id": "2401.04088",
                "title": "Mixtral of Experts",
                "abstract": (
                    "We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Each layer is composed of 8 feedforward blocks. "
                    "For every token, a router selects two experts to process the state and sum their outputs. "
                    "Even though each token has access to 47B parameters, it only uses 13B active parameters during inference."
                ),
                "authors": ["Albert Q. Jiang", "Alexandre Sablayrolles", "Antoine Roux", "Arthur Mensch", "Guillaume Lample"],
                "categories": ["cs.CL", "cs.LG", "cs.AI"],
                "year": 2024,
                "key_findings": [
                    "Demonstrated Sparse Mixture of Experts (SMoE) outperforms Llama 2 70B with 6x faster inference throughput.",
                    "Utilized top-2 routing across 8 experts per token to optimize parameter efficiency.",
                    "Natively supported 32k token context window with strong mathematical and coding reasoning."
                ]
            }
        ]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens, filtering punctuation and stopwords."""
        if not text:
            return []
        raw_words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text.lower())
        return [w for w in raw_words if len(w) > 1 and w not in STOPWORDS]

    def _build_keyword_index(self):
        """Create inverted index: word -> set of paper indices, along with token frequency profiles."""
        self.keyword_index = defaultdict(set)
        self.paper_tokens = []

        for idx, paper in enumerate(self.papers):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            categories = " ".join(paper.get("categories", []))
            authors = " ".join(paper.get("authors", []))

            findings = paper.get("key_findings", [])
            findings_str = " ".join(findings) if isinstance(findings, list) else str(findings)

            # Combined weighted text
            title_tokens = self._tokenize(title)
            abstract_tokens = self._tokenize(abstract)
            findings_tokens = self._tokenize(findings_str)
            cat_tokens = self._tokenize(categories)
            author_tokens = self._tokenize(authors)

            # All unique tokens for inverted index
            all_tokens = set(title_tokens + abstract_tokens + findings_tokens + cat_tokens + author_tokens)
            for token in all_tokens:
                self.keyword_index[token].add(idx)

            # Store token counts with weighting for paper profile
            token_counter = Counter()
            for t in title_tokens:
                token_counter[t] += 3
            for t in findings_tokens:
                token_counter[t] += 2
            for t in abstract_tokens:
                token_counter[t] += 1
            for t in cat_tokens:
                token_counter[t] += 2
            for t in author_tokens:
                token_counter[t] += 1

            self.paper_tokens.append(token_counter)

        logger.info(f"Built keyword index with {len(self.keyword_index)} distinct terms across {len(self.papers)} papers.")

    def _init_embeddings_and_llm(self):
        """Initialize HuggingFace embeddings and Google Palm / OpenAI LLMs with graceful fallbacks."""
        # Try HuggingFace Instruct Embeddings
        try:
            from langchain.embeddings import HuggingFaceInstructEmbeddings
            self.embeddings = HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-large",
                model_kwargs={"device": "cpu"}
            )
            self.embedding_model_name = "hkunlp/instructor-large"
            logger.info("Initialized HuggingFaceInstructEmbeddings (hkunlp/instructor-large).")
        except Exception as e1:
            logger.info(f"HuggingFaceInstructEmbeddings not available ({e1}). Attempting sentence-transformers fallback...")
            try:
                from langchain.embeddings.huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                logger.info("Initialized HuggingFaceEmbeddings (all-MiniLM-L6-v2).")
            except Exception as e2:
                logger.warning(f"Vector embeddings could not be initialized ({e2}). Using keyword and heuristic engine.")
                self.embeddings = None
                self.embedding_model_name = None

        # Try initializing LLM
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        if google_api_key:
            try:
                from langchain.llms import GooglePalm
                self.llm = GooglePalm(google_api_key=google_api_key, temperature=0.2)
                logger.info("Initialized GooglePalm LLM.")
            except Exception as eg:
                logger.info(f"GooglePalm not available ({eg}). Attempting Google GenerativeAI...")
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.2, max_retries=1)
                    logger.info("Initialized ChatGoogleGenerativeAI LLM.")
                except Exception as egg:
                    logger.warning(f"Google LLM init failed: {egg}")
        elif openai_api_key:
            try:
                from langchain.chat_models import ChatOpenAI
                self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2)
                logger.info("Initialized ChatOpenAI LLM.")
            except Exception as eo:
                logger.warning(f"OpenAI LLM init failed: {eo}")

    def _try_load_vector_db(self):
        """Attempt to load an existing local FAISS vector database."""
        if not self.embeddings:
            return

        resolved_db_path = self._resolve_path(self.vectordb_path)
        if resolved_db_path.exists():
            try:
                from langchain.vectorstores import FAISS
                try:
                    self.vectordb = FAISS.load_local(
                        str(resolved_db_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                except TypeError:
                    self.vectordb = FAISS.load_local(str(resolved_db_path), self.embeddings)

                self.vector_store_ready = True
                logger.info(f"Successfully loaded existing FAISS vector store from {resolved_db_path}")
            except Exception as e:
                logger.warning(f"Could not load local FAISS index ({e}). Vector DB can be built with create_paper_vector_db().")
                self.vector_store_ready = False

    def create_paper_vector_db(self) -> bool:
        """Create Document objects from papers and build/save a local FAISS vector store."""
        if not self.papers:
            logger.error("No papers loaded in engine to create vector database.")
            return False

        if not self.embeddings:
            logger.warning("Embeddings model not available. Vector database cannot be built.")
            return False

        try:
            from langchain.schema import Document
            from langchain.vectorstores import FAISS

            documents = []
            for paper in self.papers:
                p_id = paper.get("id", "")
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                authors_str = ", ".join(paper.get("authors", []))
                categories_str = ", ".join(paper.get("categories", []))
                year = paper.get("year", "N/A")

                findings = paper.get("key_findings", [])
                if isinstance(findings, list):
                    findings_str = "\n- " + "\n- ".join(findings)
                else:
                    findings_str = str(findings)

                content = (
                    f"Paper ID: {p_id}\n"
                    f"Title: {title}\n"
                    f"Authors: {authors_str}\n"
                    f"Year: {year}\n"
                    f"Categories: {categories_str}\n"
                    f"Key Findings:\n{findings_str}\n\n"
                    f"Abstract:\n{abstract}"
                )

                metadata = {
                    "id": str(p_id),
                    "title": title,
                    "year": year,
                    "categories": categories_str,
                    "authors": authors_str
                }
                documents.append(Document(page_content=content, metadata=metadata))

            logger.info(f"Building FAISS index for {len(documents)} documents...")
            self.vectordb = FAISS.from_documents(documents=documents, embedding=self.embeddings)

            # Determine save path
            save_path = Path(self.vectordb_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.vectordb.save_local(str(save_path))
            self.vector_store_ready = True
            logger.info(f"FAISS vector store saved successfully to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating FAISS vector database: {e}", exc_info=True)
            return False

    def search_papers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search papers using vector similarity if available, with keyword matching fallback."""
        if not query or not query.strip():
            return [dict(p, relevance_score=1.0, match_source="all") for p in self.papers[:top_k]]

        query_cleaned = query.strip()

        # 1. Try vector store search if ready
        if self.vector_store_ready and self.vectordb:
            try:
                docs_with_scores = self.vectordb.similarity_search_with_score(query_cleaned, k=top_k)
                results = []
                for doc, score in docs_with_scores:
                    paper_id = doc.metadata.get("id")
                    paper = self.get_paper_by_id(paper_id)
                    if paper:
                        # Convert L2 distance or score to normalized 0-1 relevance
                        relevance = max(0.0, min(1.0, 1.0 / (1.0 + float(score))))
                        paper_copy = dict(paper)
                        paper_copy["relevance_score"] = round(relevance, 4)
                        paper_copy["match_source"] = "vector"
                        results.append(paper_copy)

                if results:
                    return results
            except Exception as e:
                logger.warning(f"Vector search failed ({e}), falling back to keyword search.")

        # 2. Keyword-based matching
        query_tokens = self._tokenize(query_cleaned)
        if not query_tokens:
            # Fallback: simple substring search
            sub_results = []
            q_lower = query_cleaned.lower()
            for p in self.papers:
                if q_lower in p.get("title", "").lower() or q_lower in p.get("abstract", "").lower():
                    copy_p = dict(p)
                    copy_p["relevance_score"] = 0.75
                    copy_p["match_source"] = "substring"
                    sub_results.append(copy_p)
            return sub_results[:top_k] if sub_results else [dict(p, relevance_score=0.1, match_source="fallback") for p in self.papers[:top_k]]

        matched_indices: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            target_indices = self.keyword_index.get(token, set())
            for idx in target_indices:
                # TF weighting in this paper
                tf = self.paper_tokens[idx].get(token, 1)
                # IDF weighting
                doc_freq = len(self.keyword_index.get(token, []))
                idf = math.log(1.0 + (len(self.papers) / (1.0 + doc_freq)))
                matched_indices[idx] += tf * idf

            # Also check for partial substring matches in token stems
            for indexed_token, paper_idxs in self.keyword_index.items():
                if token in indexed_token and indexed_token != token:
                    for idx in paper_idxs:
                        matched_indices[idx] += 0.5

        if not matched_indices:
            # Fallback: fuzzy overlap
            for idx, p in enumerate(self.papers):
                text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
                for q_t in query_tokens:
                    if q_t in text:
                        matched_indices[idx] += 1.0

        if not matched_indices:
            return [dict(p, relevance_score=0.0, match_source="none") for p in self.papers[:top_k]]

        max_score = max(matched_indices.values()) if matched_indices else 1.0
        sorted_indices = sorted(matched_indices.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, raw_score in sorted_indices[:top_k]:
            paper = dict(self.papers[idx])
            paper["relevance_score"] = round(raw_score / max_score if max_score > 0 else 1.0, 4)
            paper["match_source"] = "keyword"
            results.append(paper)

        return results

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Find and return paper dictionary by exact or normalized ID."""
        if not paper_id:
            return None

        clean_id = str(paper_id).strip().lower().replace("arxiv:", "").replace("v1", "").replace("v2", "")
        for p in self.papers:
            curr_id = str(p.get("id", "")).strip().lower().replace("arxiv:", "").replace("v1", "").replace("v2", "")
            if curr_id == clean_id:
                return p

        # Check if partial ID matches
        for p in self.papers:
            if clean_id in str(p.get("id", "")).lower():
                return p

        return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Safely execute an LLM prompt if configured."""
        if not self.llm:
            return None
        try:
            if hasattr(self.llm, "predict"):
                return self.llm.predict(prompt).strip()
            elif hasattr(self.llm, "invoke"):
                res = self.llm.invoke(prompt)
                return res.content if hasattr(res, "content") else str(res).strip()
            elif callable(self.llm):
                res = self.llm(prompt)
                return str(res).strip()
        except Exception as e:
            logger.warning(f"LLM generation failed ({e}), using deterministic template fallback.")
        return None

    def summarize_paper(self, paper_id: str, level: str = "executive") -> str:
        """Summarize a paper at different levels: 'executive', 'technical', 'layperson'."""
        paper = self.get_paper_by_id(paper_id)
        if not paper:
            return f"Error: Paper ID '{paper_id}' not found in the arXiv knowledge base."

        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "")
        year = paper.get("year", "N/A")
        authors = ", ".join(paper.get("authors", []))
        categories = ", ".join(paper.get("categories", []))
        findings = paper.get("key_findings", [])
        findings_text = "\n- " + "\n- ".join(findings) if isinstance(findings, list) else str(findings)

        level_norm = level.lower().strip()

        # Try LLM generation if available
        if self.llm:
            prompt = (
                f"You are a leading AI domain research expert. Summarize the following arXiv paper at the '{level_norm}' level.\n\n"
                f"Paper ID: {paper.get('id')}\n"
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Year: {year}\n"
                f"Categories: {categories}\n"
                f"Key Findings:\n{findings_text}\n"
                f"Abstract:\n{abstract}\n\n"
                f"Formatting Instructions:\n"
            )
            if level_norm == "executive":
                prompt += "- Provide a 2-3 sentence high-level executive briefing highlighting the core problem, breakthrough, and business/research value.\n"
            elif level_norm == "layperson":
                prompt += "- Explain what the paper does in clear, plain, everyday English with no heavy math or obscure jargon. Use analogies if helpful.\n"
            else:
                prompt += "- Provide an in-depth technical breakdown covering architecture, core mechanisms, mathematical/algorithmic formulation, and empirical benchmarks.\n"

            llm_summary = self._call_llm(prompt)
            if llm_summary:
                return f"### Summary ({level_norm.capitalize()} Level): {title}\n\n{llm_summary}"

        # Deterministic Template-based Summarization
        if level_norm == "executive":
            first_finding = findings[0] if isinstance(findings, list) and findings else "Introduced fundamental algorithmic improvements."
            summary = (
                f"### Executive Summary: {title} ({year})\n\n"
                f"**Paper ID**: `{paper.get('id')}` | **Field**: {categories}\n\n"
                f"**Strategic Overview**: {title} addressed critical computational and representational bottlenecks in AI by "
                f"proposing novel architectures that drastically improved efficiency and scalability.\n\n"
                f"**Key Highlights & Impact**:\n"
                f"- **Core Innovation**: {first_finding}\n"
                f"- **Significance**: Provided foundational advancements enabling scalable training and high-accuracy downstream deployment across the {categories} domain.\n\n"
                f"**Key Findings**:{findings_text}"
            )
            return summary

        elif level_norm == "layperson":
            summary = (
                f"### Plain English Guide: {title}\n\n"
                f"**What is this paper about?**\n"
                f"Imagine trying to teach computers to understand human language, images, or logic. This paper, written in {year}, "
                f"introduces a smarter way for artificial intelligence to learn and solve complex problems faster.\n\n"
                f"**The Big Problem It Solves**:\n"
                f"Traditional systems were either too slow, required massive amounts of memory, or struggled to remember long context and details.\n\n"
                f"**How It Works (In Simple Terms)**:\n"
                f"{abstract}\n\n"
                f"**Why It Matters To You**:\n"
                f"The breakthroughs in this research directly power the next generation of AI assistants, search engines, and smart software we use today."
            )
            return summary

        else:  # 'technical' level
            summary = (
                f"### Technical Research Breakdown: {title}\n\n"
                f"- **ArXiv ID**: `{paper.get('id')}`\n"
                f"- **Authors**: {authors}\n"
                f"- **Primary Categories**: {categories}\n"
                f"- **Publication Year**: {year}\n\n"
                f"#### Abstract\n{abstract}\n\n"
                f"#### Core Architecture & Key Contributions\n{findings_text}\n\n"
                f"#### Methodological Context\n"
                f"This work directly advances sequence transduction and representation modeling within `{categories}`. "
                f"By eliminating structural bottlenecks and optimizing computational graphs, it sets new empirical standards for modern foundation models."
            )
            return summary

    def explain_concept(self, concept: str) -> str:
        """Explain a technical concept using knowledge compiled from the paper corpus."""
        if not concept or not concept.strip():
            return "Please provide a concept to explain."

        concept_clean = concept.strip()
        matching_papers = self.search_papers(concept_clean, top_k=4)

        # Context compilation
        corpus_context = []
        for p in matching_papers:
            findings_str = "; ".join(p.get("key_findings", [])) if isinstance(p.get("key_findings"), list) else str(p.get("key_findings", ""))
            corpus_context.append(
                f"- Paper '{p.get('title')}' ({p.get('year')}): {findings_str}. Abstract excerpt: {p.get('abstract', '')[:200]}..."
            )
        context_str = "\n".join(corpus_context)

        # LLM explanation
        if self.llm and matching_papers:
            prompt = (
                f"You are an expert AI research scientist. Explain the concept '{concept_clean}' clearly and comprehensively "
                f"using insights from the following related arXiv papers:\n\n"
                f"{context_str}\n\n"
                f"Structure your explanation with:\n"
                f"1. Definition & Core Intuition\n"
                f"2. How It Works Mechanically\n"
                f"3. Key Papers & Milestones in Corpus\n"
                f"4. Real-World Applications & Advantages\n"
            )
            explanation = self._call_llm(prompt)
            if explanation:
                return f"## Concept Explanation: {concept_clean}\n\n{explanation}"

        # Heuristic / Template Explanation
        ref_titles = [f"`{p.get('title')}` ({p.get('year')})" for p in matching_papers[:3]]
        refs_str = ", ".join(ref_titles) if ref_titles else "the literature"

        explanation = (
            f"## Concept Explanation: {concept_clean}\n\n"
            f"### 1. Definition & Intuition\n"
            f"**{concept_clean}** represents a pivotal paradigm in modern artificial intelligence and machine learning. "
            f"Within our arXiv research corpus, it is primarily discussed in the context of {refs_str}.\n\n"
            f"### 2. Mechanics & Key Principles\n"
            f"Based on the indexed papers, {concept_clean} addresses fundamental trade-offs between model expressivity, "
            f"computational efficiency, and generalization quality. Key mechanisms include:\n"
        )

        for p in matching_papers[:3]:
            findings = p.get("key_findings", [])
            finding_snippet = findings[0] if isinstance(findings, list) and findings else p.get("abstract", "")[:150]
            explanation += f"- **From {p.get('title')} ({p.get('year')})**: {finding_snippet}\n"

        explanation += (
            f"\n### 3. Relevant Papers in Knowledge Base\n"
        )
        for p in matching_papers:
            explanation += f"- **[{p.get('id')}] {p.get('title')}** ({p.get('year')}) - *{', '.join(p.get('categories', []))}*\n"

        return explanation

    def get_related_papers(self, paper_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find papers with overlapping categories, authors, or keyword similarity."""
        target_paper = self.get_paper_by_id(paper_id)
        if not target_paper:
            return []

        target_idx = None
        for i, p in enumerate(self.papers):
            if str(p.get("id")) == str(target_paper.get("id")):
                target_idx = i
                break

        target_cats = set(target_paper.get("categories", []))
        target_tokens = self.paper_tokens[target_idx] if target_idx is not None else Counter(self._tokenize(target_paper.get("title", "")))

        scores: List[Tuple[int, float, List[str]]] = []

        for idx, paper in enumerate(self.papers):
            if idx == target_idx or str(paper.get("id")) == str(target_paper.get("id")):
                continue

            # Category Jaccard similarity
            paper_cats = set(paper.get("categories", []))
            cat_overlap = len(target_cats.intersection(paper_cats))
            cat_union = len(target_cats.union(paper_cats))
            cat_score = cat_overlap / cat_union if cat_union > 0 else 0.0

            # Token overlap similarity (Cosine / Dot product of normalized term counters)
            shared_terms = set(target_tokens.keys()).intersection(set(self.paper_tokens[idx].keys()))
            token_score = sum(target_tokens[t] * self.paper_tokens[idx][t] for t in shared_terms)
            norm_factor = (math.sqrt(sum(v**2 for v in target_tokens.values())) *
                           math.sqrt(sum(v**2 for v in self.paper_tokens[idx].values()))) + 1e-6
            cos_score = token_score / norm_factor

            # Combined similarity
            combined = (cat_score * 0.4) + (cos_score * 0.6)
            shared_cat_list = list(target_cats.intersection(paper_cats))
            scores.append((idx, combined, shared_cat_list))

        scores.sort(key=lambda x: x[1], reverse=True)

        related = []
        for idx, sim, shared_c in scores[:top_k]:
            p_dict = dict(self.papers[idx])
            p_dict["similarity_score"] = round(sim, 4)
            p_dict["shared_categories"] = shared_c
            related.append(p_dict)

        return related

    def generate_concept_graph_data(self, query: str = "") -> Dict[str, Any]:
        """Generate graph nodes and co-occurrence edges for concept visualization."""
        selected_papers = self.search_papers(query, top_k=10) if query and query.strip() else self.papers

        # Concept vocabulary / domain entities
        domain_concepts = [
            "Transformer", "Attention", "Self-Attention", "BERT", "GPT", "Few-Shot Learning",
            "Residual Learning", "ResNet", "Skip Connections", "LoRA", "Parameter-Efficient Fine-Tuning",
            "FlashAttention", "Memory Optimization", "Diffusion Models", "Denoising", "Generative Modeling",
            "Retrieval-Augmented Generation", "Dense Retrieval", "Reinforcement Learning", "RLHF",
            "Human Feedback", "Contrastive Learning", "CLIP", "Multimodal", "Direct Preference Optimization",
            "DPO", "Mixture of Experts", "Sparse Routing", "Large Language Models", "Computer Vision"
        ]

        concept_paper_map: Dict[str, Set[str]] = defaultdict(set)
        concept_categories: Dict[str, Counter] = defaultdict(Counter)

        for paper in selected_papers:
            p_id = str(paper.get("id"))
            p_text = f"{paper.get('title', '')} {paper.get('abstract', '')} {' '.join(paper.get('key_findings', []))}".lower()
            cats = paper.get("categories", ["cs.AI"])
            main_cat = cats[0] if cats else "cs.AI"

            for concept in domain_concepts:
                c_lower = concept.lower()
                if c_lower in p_text:
                    concept_paper_map[concept].add(p_id)
                    concept_categories[concept][main_cat] += 1

        # Also add paper categories as overarching hub nodes
        for paper in selected_papers:
            for cat in paper.get("categories", []):
                concept_paper_map[cat].add(str(paper.get("id")))
                concept_categories[cat][cat] += 1

        # Filter concepts that appeared in at least 1 paper
        active_concepts = [c for c, p_set in concept_paper_map.items() if len(p_set) > 0]

        nodes = []
        for concept in active_concepts:
            papers_count = len(concept_paper_map[concept])
            dominant_cat = concept_categories[concept].most_common(1)[0][0] if concept_categories[concept] else "cs.AI"
            nodes.append({
                "id": concept,
                "label": concept,
                "size": max(10, min(50, papers_count * 8 + 10)),
                "paper_count": papers_count,
                "category": dominant_cat,
                "category_label": CATEGORY_MAP.get(dominant_cat, dominant_cat)
            })

        # Calculate co-occurrence edges
        edges = []
        for i in range(len(active_concepts)):
            for j in range(i + 1, len(active_concepts)):
                c1 = active_concepts[i]
                c2 = active_concepts[j]
                common_papers = concept_paper_map[c1].intersection(concept_paper_map[c2])
                if common_papers:
                    edges.append({
                        "source": c1,
                        "target": c2,
                        "weight": len(common_papers),
                        "shared_papers": list(common_papers)
                    })

        return {
            "query": query,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes,
            "edges": edges
        }

    def answer_question(self, question: str, conversation_history: Optional[List[str]] = None) -> Dict[str, Any]:
        """Answer a domain question using paper knowledge base with source attribution and concept links."""
        if not question or not question.strip():
            return {
                "answer": "Please ask a question about AI/CS research papers.",
                "sources": [],
                "related_papers": [],
                "concepts": []
            }

        # Contextual search considering recent query history
        search_query = question
        if conversation_history:
            recent_turns = " ".join(conversation_history[-2:])
            search_query = f"{question} {recent_turns}"

        top_papers = self.search_papers(search_query, top_k=4)

        # Build context from top papers
        context_blocks = []
        concepts_identified = set()
        for p in top_papers:
            p_id = p.get("id")
            title = p.get("title")
            year = p.get("year")
            findings = "; ".join(p.get("key_findings", [])) if isinstance(p.get("key_findings"), list) else str(p.get("key_findings"))
            abstract = p.get("abstract", "")
            context_blocks.append(f"[{p_id}] '{title}' ({year}):\nKey Findings: {findings}\nAbstract: {abstract}")

            for word in self._tokenize(title):
                if len(word) > 4:
                    concepts_identified.add(word.capitalize())

        context_text = "\n\n".join(context_blocks)

        # Generate answer using LLM if available
        llm_answer = None
        if self.llm:
            history_text = ""
            if conversation_history:
                history_text = "Prior Conversation History:\n" + "\n".join(conversation_history[-4:]) + "\n\n"

            prompt = (
                f"You are a domain expert research assistant specializing in Computer Science and AI arXiv papers.\n"
                f"{history_text}"
                f"Context from arXiv papers:\n{context_text}\n\n"
                f"Question: {question}\n\n"
                f"Instructions:\n"
                f"- Answer the question thoroughly and accurately based on the paper context provided.\n"
                f"- Explicitly cite papers by ID or title when referencing methods or benchmarks.\n"
                f"- If the answer is not fully in the context, synthesize the most accurate technical explanation possible based on verified CS fundamentals."
            )
            llm_answer = self._call_llm(prompt)

        # Deterministic / Heuristic Answer if LLM not present or failed
        if not llm_answer:
            if not top_papers or top_papers[0].get("relevance_score", 0.0) < 0.05:
                answer_body = (
                    f"I could not locate specific papers in the current arXiv index directly addressing: '{question}'. "
                    f"Our database currently specializes in Foundation Models, Attention, Transformers, Diffusion, and Alignment."
                )
            else:
                p_top = top_papers[0]
                findings_top = p_top.get("key_findings", [])
                finding_str = findings_top[0] if isinstance(findings_top, list) and findings_top else p_top.get("abstract", "")

                answer_body = (
                    f"Based on **{p_top.get('title')}** ({p_top.get('year')}) [`arXiv:{p_top.get('id')}`]:\n\n"
                    f"- **Key Insight**: {finding_str}\n\n"
                    f"**Technical Details from Knowledge Base**:\n"
                )
                for p in top_papers[:3]:
                    f_list = p.get("key_findings", [])
                    f_text = f_list[0] if isinstance(f_list, list) and f_list else p.get("abstract", "")[:120] + "..."
                    answer_body += f"- **{p.get('title')}** ({p.get('year')}): {f_text}\n"

                answer_body += f"\n*Referenced from {len(top_papers)} related research publications in the corpus.*"
        else:
            answer_body = llm_answer

        sources = [
            {
                "id": p.get("id"),
                "title": p.get("title"),
                "year": p.get("year"),
                "categories": p.get("categories", []),
                "relevance_score": p.get("relevance_score", 1.0)
            }
            for p in top_papers
        ]

        return {
            "answer": answer_body,
            "sources": sources,
            "related_papers": [p.get("title") for p in top_papers],
            "concepts": list(concepts_identified)[:8]
        }

    def get_field_statistics(self) -> Dict[str, Any]:
        """Return analytics on total papers, categories distribution, year distribution, and top authors."""
        categories_count = Counter()
        year_count = Counter()
        authors_count = Counter()
        total_findings = 0

        for paper in self.papers:
            for cat in paper.get("categories", []):
                categories_count[cat] += 1

            year = paper.get("year", "Unknown")
            year_count[year] += 1

            for author in paper.get("authors", []):
                authors_count[author] += 1

            findings = paper.get("key_findings", [])
            total_findings += len(findings) if isinstance(findings, list) else 1

        avg_findings = round(total_findings / len(self.papers), 2) if self.papers else 0.0

        return {
            "total_papers": len(self.papers),
            "categories_distribution": dict(categories_count.most_common()),
            "year_distribution": dict(sorted(year_count.items(), key=lambda x: str(x[0]))),
            "top_authors": authors_count.most_common(10),
            "avg_key_findings_per_paper": avg_findings,
            "category_labels": CATEGORY_MAP
        }

    def get_trending_topics(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Analyze titles, abstracts, and key findings for most frequent meaningful terms and phrases."""
        term_counter = Counter()

        for paper in self.papers:
            title_tokens = self._tokenize(paper.get("title", ""))
            findings_str = " ".join(paper.get("key_findings", [])) if isinstance(paper.get("key_findings"), list) else str(paper.get("key_findings", ""))
            findings_tokens = self._tokenize(findings_str)

            # Unigrams
            for t in title_tokens + findings_tokens:
                if len(t) > 3:
                    term_counter[t.capitalize()] += 1

            # Bigrams
            tokens = title_tokens + findings_tokens
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                if len(w1) > 2 and len(w2) > 2:
                    bigram = f"{w1.capitalize()} {w2.capitalize()}"
                    term_counter[bigram] += 2

        return term_counter.most_common(top_n)

    def get_status(self) -> Dict[str, Any]:
        """Return engine operational status and configuration."""
        return {
            "status": "ready" if self.papers else "empty",
            "total_papers": len(self.papers),
            "vector_store_ready": self.vector_store_ready,
            "embedding_model": self.embedding_model_name,
            "llm_available": self.llm is not None,
            "dataset_path": str(self.dataset_path),
            "vectordb_path": str(self.vectordb_path),
            "keyword_index_size": len(self.keyword_index)
        }


if __name__ == "__main__":
    print("=" * 80)
    print("         ArXiv Expert Engine - Domain Intelligence Demo")
    print("=" * 80)

    # 1. Initialize Engine
    engine = ArXivExpertEngine()
    status = engine.get_status()
    print(f"\n[Status]: {status}")

    # 2. Field Statistics
    stats = engine.get_field_statistics()
    print(f"\n[Field Statistics]:")
    print(f"  - Total Papers: {stats['total_papers']}")
    print(f"  - Categories: {stats['categories_distribution']}")
    print(f"  - Years: {stats['year_distribution']}")
    print(f"  - Top 3 Authors: {stats['top_authors'][:3]}")

    # 3. Paper Search
    query = "attention transformer"
    print(f"\n[Search Query]: '{query}'")
    search_results = engine.search_papers(query, top_k=2)
    for res in search_results:
        print(f"  -> [{res.get('id')}] {res.get('title')} (Score: {res.get('relevance_score')}, Match: {res.get('match_source')})")

    # 4. Multi-Level Summaries
    paper_id = "1706.03762"
    print(f"\n[Executive Summary for {paper_id}]:")
    print(engine.summarize_paper(paper_id, level="executive"))

    print(f"\n[Layperson Summary for {paper_id}]:")
    print(engine.summarize_paper(paper_id, level="layperson"))

    # 5. Concept Explanation
    concept = "Retrieval-Augmented Generation"
    print(f"\n[Explain Concept: '{concept}']:")
    print(engine.explain_concept(concept))

    # 6. Question Answering
    question = "How does LoRA reduce fine-tuning memory overhead?"
    print(f"\n[Q&A]: '{question}'")
    qa_result = engine.answer_question(question)
    print(f"Answer:\n{qa_result['answer']}")
    print(f"Sources: {[s['title'] for s in qa_result['sources']]}")

    # 7. Related Papers
    print(f"\n[Related Papers for {paper_id}]:")
    related = engine.get_related_papers(paper_id, top_k=2)
    for rel in related:
        print(f"  -> [{rel.get('id')}] {rel.get('title')} (Similarity: {rel.get('similarity_score')})")

    # 8. Concept Graph Data
    print(f"\n[Concept Graph Generation]:")
    graph = engine.generate_concept_graph_data("language models")
    print(f"  - Generated {graph['total_nodes']} nodes and {graph['total_edges']} edges.")

    # 9. Trending Topics
    print(f"\n[Trending Topics in Corpus]:")
    for topic, count in engine.get_trending_topics(top_n=5):
        print(f"  - {topic}: {count} occurrences")

    print("\n" + "=" * 80)
    print("Demo completed successfully.")
    print("=" * 80)
