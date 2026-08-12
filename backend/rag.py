"""
miniVoxSetu — RAG (Retrieval-Augmented Generation) Engine

Production upgrade from MVP:
  Phase 2: BM25 + Vector Hybrid Search with Reciprocal Rank Fusion (RRF)
  Phase 3: Qdrant Vector Database (with SimpleVectorStore fallback for local dev)
  Phase 4: Dynamic document ingestion support (see ingest.py)

ARCHITECTURE:
  Query → [BM25 Keyword Search] ─┐
                                  ├─→ RRF Fusion → Top-N Results
  Query → [Vector Similarity]  ──┘

EMBEDDING MODEL:
  sentence-transformers/all-MiniLM-L6-v2 running locally on CPU.
  Local embedding takes ~5ms on CPU — a 30x speedup over cloud APIs.

VECTOR DATABASE MODES (set via VECTOR_DB_MODE env var):
  "memory"  → In-memory SimpleVectorStore (default, no external deps)
  "qdrant"  → Qdrant via qdrant-client (Docker or Qdrant Cloud)
"""

import os
import hashlib
import numpy as np
from typing import Optional

# ============================================================
# DEPENDENCY IMPORTS (graceful fallback for each)
# ============================================================

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"[RAG] Loaded local embedding model: all-MiniLM-L6-v2 (384 dims)")
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    _embedding_model = None
    EMBEDDINGS_AVAILABLE = False
    print(f"[RAG] ⚠️ sentence-transformers not installed: {e}. RAG will be disabled.")

# BM25 keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
    print("[RAG] ✅ BM25 keyword search available (rank_bm25)")
except ImportError:
    BM25_AVAILABLE = False
    print("[RAG] ⚠️ rank_bm25 not installed. Using vector-only retrieval.")

# Qdrant vector database
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue
    )
    QDRANT_AVAILABLE = True
    print("[RAG] ✅ Qdrant client available")
except ImportError:
    QDRANT_AVAILABLE = False
    print("[RAG] ⚠️ qdrant-client not installed. Using in-memory vector store.")

# Vector dimension for all-MiniLM-L6-v2
VECTOR_DIM = 384
COLLECTION_NAME = "minivoxsetu_knowledge"


# ============================================================
# SIMPLE VECTOR STORE (in-memory fallback — original MVP code)
# ============================================================

class SimpleVectorStore:
    """
    In-memory brute-force cosine similarity search.
    Used as fallback when Qdrant is not available (local development).

    Production equivalent: Qdrant HNSW index with payload filtering.
    """

    def __init__(self):
        self.documents = []      # List of document strings
        self.embeddings = []     # List of embedding vectors (numpy arrays)
        self.ids = []            # Document IDs
        self._hashes = set()     # SHA256 hashes for deduplication

    def add(self, ids: list[str], embeddings: list, documents: list[str]):
        """Store documents alongside their embeddings (deduplication by content hash)."""
        for doc_id, embedding, doc in zip(ids, embeddings, documents):
            doc_hash = hashlib.sha256(doc.encode()).hexdigest()
            if doc_hash in self._hashes:
                continue  # Skip duplicate content
            self._hashes.add(doc_hash)
            self.ids.append(doc_id)
            self.documents.append(doc)
            self.embeddings.append(np.array(embedding))

    def query(self, query_embedding, n_results: int = 2) -> list[tuple[str, str, float]]:
        """
        Find most similar documents by cosine similarity.
        Returns list of (doc_id, document_text, similarity_score).
        """
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding)
        similarities = []
        for i, doc_vec in enumerate(self.embeddings):
            cos_sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-10
            )
            similarities.append((i, float(cos_sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in similarities[:n_results]:
            results.append((self.ids[idx], self.documents[idx], score))
        return results

    def get_all_documents(self) -> list[tuple[str, str]]:
        """Return all (id, document) pairs for BM25 indexing."""
        return list(zip(self.ids, self.documents))

    def count(self) -> int:
        return len(self.documents)


# ============================================================
# QDRANT VECTOR STORE (production — Phase 3)
# ============================================================

class QdrantVectorStore:
    """
    Qdrant-backed vector store with same interface as SimpleVectorStore.
    Supports both local Docker and Qdrant Cloud connections.

    ENV VARS:
      QDRANT_URL      → Qdrant server URL (default: http://localhost:6333)
      QDRANT_API_KEY   → API key for Qdrant Cloud (optional for local Docker)
    """

    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print(f"[RAG] Connected to Qdrant Cloud: {qdrant_url}")
        else:
            self.client = QdrantClient(url=qdrant_url)
            print(f"[RAG] Connected to Qdrant (local): {qdrant_url}")

        # Create collection if it doesn't exist
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_DIM,
                        distance=Distance.COSINE
                    )
                )
                print(f"[RAG] Created Qdrant collection: {COLLECTION_NAME}")
            else:
                print(f"[RAG] Using existing Qdrant collection: {COLLECTION_NAME}")
        except Exception as e:
            print(f"[RAG] ⚠️ Qdrant collection setup error: {e}")
            raise

        self._doc_cache = {}  # Local cache: id → document text

    def add(self, ids: list[str], embeddings: list, documents: list[str]):
        """Upsert documents into Qdrant collection."""
        points = []
        for doc_id, embedding, doc in zip(ids, embeddings, documents):
            # Use hash of doc_id as integer point ID (Qdrant requires int or UUID)
            point_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:15], 16)
            points.append(PointStruct(
                id=point_id,
                vector=embedding if isinstance(embedding, list) else embedding.tolist(),
                payload={
                    "doc_id": doc_id,
                    "text": doc,
                    "content_hash": hashlib.sha256(doc.encode()).hexdigest()
                }
            ))
            self._doc_cache[doc_id] = doc

        if points:
            # Batch upsert (Qdrant handles dedup by point ID)
            batch_size = 100
            for i in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points[i:i + batch_size]
                )
            print(f"[RAG] Upserted {len(points)} points to Qdrant")

    def query(self, query_embedding, n_results: int = 2) -> list[tuple[str, str, float]]:
        """Search Qdrant for similar documents. Returns list of (doc_id, text, score)."""
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding if isinstance(query_embedding, list)
                    else query_embedding.tolist(),
                limit=n_results
            )
            return [
                (hit.payload.get("doc_id", ""), hit.payload.get("text", ""), hit.score)
                for hit in results
            ]
        except Exception as e:
            print(f"[RAG] Qdrant search error: {e}")
            return []

    def get_all_documents(self) -> list[tuple[str, str]]:
        """Retrieve all documents from Qdrant for BM25 indexing."""
        try:
            # Scroll through all points
            all_docs = []
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                for point in results:
                    doc_id = point.payload.get("doc_id", "")
                    text = point.payload.get("text", "")
                    all_docs.append((doc_id, text))
                if offset is None:
                    break
            return all_docs
        except Exception as e:
            print(f"[RAG] Qdrant scroll error: {e}")
            return list(self._doc_cache.items())

    def count(self) -> int:
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return info.points_count
        except Exception:
            return 0


# ============================================================
# BM25 KEYWORD INDEX (Phase 2)
# ============================================================

class BM25Index:
    """
    BM25Okapi keyword search index.
    Tokenizes documents at index time and provides ranked keyword retrieval.

    WHY BM25 alongside vector search:
    - Vector search captures MEANING ("How do I open an account?" ≈ "account opening process")
    - BM25 captures EXACT TERMS ("IFSC", "NEFT", "TDS 15G", specific product names)
    - Hybrid = best of both worlds
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.doc_ids: list[str] = []
        self.documents: list[str] = []

    def build_index(self, doc_pairs: list[tuple[str, str]]):
        """Build BM25 index from (doc_id, document_text) pairs."""
        if not BM25_AVAILABLE or not doc_pairs:
            return

        self.doc_ids = [doc_id for doc_id, _ in doc_pairs]
        self.documents = [doc for _, doc in doc_pairs]

        # Tokenize: simple whitespace + lowercasing
        # For Indian banking domain, this works well since key terms
        # (NEFT, IFSC, UPI, Aadhaar) are distinct tokens
        tokenized = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[RAG] BM25 index built with {len(self.documents)} documents")

    def search(self, query: str, n_results: int = 10) -> list[str]:
        """
        Search BM25 index. Returns ranked list of doc_ids.
        """
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices sorted by BM25 score (descending)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.doc_ids[i] for i in ranked_indices[:n_results]]


# ============================================================
# RECIPROCAL RANK FUSION (Phase 2)
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[str]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ 1/(k + rank(d, r)) for each ranking r
    - k=60 is the standard constant (smooths score distribution)
    - Operates on RANK POSITIONS, not raw scores (avoids score normalization issues)

    Returns: list of doc_ids sorted by fused score (highest first).
    """
    fused_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return [
        doc_id for doc_id, _
        in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ============================================================
# FAQ KNOWLEDGE BASE (seed data — migrated to file in Phase 4)
# ============================================================

# WHY KEPT: These serve as fallback seed data if no files exist in knowledge/ directory.
# In production, documents are loaded from backend/knowledge/ via ingest.py.
FAQ_DOCUMENTS = [
    # --- Account Types ---
    "NeoBank offers three account types: Basic Savings (zero balance, free), "
    "Premium Savings (₹500/month, higher limits and priority support), and "
    "Current Account (₹1,000/month, best for businesses with unlimited transactions and invoicing).",

    # --- Account Opening ---
    "To open a NeoBank account, you need Aadhaar card, PAN card, and a selfie for Video KYC. "
    "The entire process is completed online in under 5 minutes. No branch visit is required. "
    "You must be at least 18 years old and an Indian resident.",

    # --- Interest Rates ---
    "NeoBank savings account interest rates: Basic members earn 3.5% per annum, "
    "Premium members earn 6.5% per annum. Interest is compounded quarterly and "
    "credited to your account on the last day of each quarter.",

    # --- UPI ---
    "NeoBank fully supports UPI payments through our app. You can send up to ₹1 lakh per "
    "transaction and ₹2 lakh per day via UPI. UPI payments are instant, available 24x7, "
    "and completely free of charge. We support all UPI apps including Google Pay, PhonePe, and Paytm.",

    # --- NEFT/IMPS/RTGS ---
    "For fund transfers: NEFT is free and settles in 30-minute batches (available 24x7). "
    "IMPS is instant with a ₹5 fee for transfers above ₹1 lakh. "
    "RTGS is for amounts above ₹2 lakh, settles in real-time, and has a ₹20 fee.",

    # --- Fixed Deposits ---
    "NeoBank Fixed Deposit rates: 7.1% for 1 year, 7.5% for 2 years, and 7.8% for 3 years. "
    "Senior citizens get an additional 0.5% on all tenures. Minimum FD amount is ₹10,000. "
    "Premature withdrawal is allowed with a 1% penalty on the applicable rate.",

    # --- Debit Card ---
    "NeoBank issues a free RuPay debit card with Basic accounts and a Visa Platinum card "
    "with Premium accounts. Daily ATM withdrawal limit is ₹25,000 for Basic and ₹1 lakh for "
    "Premium. Card replacement is free and ships within 2 business days.",

    # --- Lost/Stolen Card ---
    "If your NeoBank debit card is lost or stolen, immediately block it in the app under "
    "Settings → Cards → Block Card. You can also call our 24x7 helpline at 1800-NEO-BANK (toll-free). "
    "A replacement card is issued automatically and reaches you within 2 working days at no extra charge.",

    # --- Loans ---
    "NeoBank offers personal loans from ₹50,000 to ₹25 lakh at interest rates starting "
    "from 10.5% per annum. Loan approval takes just 2 minutes for pre-approved customers. "
    "EMI options range from 6 to 60 months. No prepayment or foreclosure charges after 6 months.",

    # --- EMI Queries ---
    "To check your loan EMI details, go to the app → Loans → Active Loans. You'll see your "
    "EMI amount, next due date, remaining tenure, and total outstanding principal. "
    "You can also set up auto-debit for EMI payments to avoid late fees.",

    # --- Customer Support ---
    "NeoBank customer support: 24x7 in-app chat (average response time: 30 seconds), "
    "phone support at 1800-NEO-BANK (toll-free, 8 AM to 10 PM IST daily), "
    "email at support@neobank.in (response within 4 hours). Premium members get a dedicated "
    "relationship manager.",

    # --- Complaints ---
    "To file a complaint, go to app → Help → Raise a Complaint. Your complaint will be "
    "assigned a ticket number and resolved within 48 hours. If not resolved, you can escalate "
    "to the Nodal Officer via email at nodal@neobank.in or the Banking Ombudsman.",

    # --- KYC ---
    "NeoBank offers instant Video KYC from your phone. You need your Aadhaar and PAN handy. "
    "The video call takes about 3 minutes. If your KYC is pending or expired, some features "
    "like fund transfers and UPI will be temporarily restricted until KYC is completed.",

    # --- International Transactions ---
    "NeoBank Visa Platinum card supports international transactions. Daily international "
    "spending limit is ₹5 lakh. Foreign currency markup is 1.5% (one of the lowest in India). "
    "You need to enable international transactions in the app under Settings → Cards → International Usage.",

    # --- Overdraft Protection ---
    "NeoBank does not charge overdraft fees. If your balance is insufficient, the transaction "
    "is simply declined. Premium members can opt into Overdraft Protection which links to their "
    "Fixed Deposit — the bank auto-liquidates the FD to cover the shortfall.",

    # --- Cheque Book ---
    "NeoBank offers a free cheque book (20 leaves) for Current Account holders. For Savings "
    "account holders, a cheque book can be requested at ₹100 for 25 leaves. "
    "Request via app → Services → Order Cheque Book. Delivery takes 5-7 working days.",

    # --- Mobile App Features ---
    "NeoBank's mobile app supports biometric login (Face ID, fingerprint), real-time push "
    "notifications for every transaction, AI-powered spending insights, UPI payments, "
    "bill payments (electricity, mobile, DTH), and scan-and-pay via QR code.",

    # --- Transaction Limits ---
    "NeoBank daily transaction limits: UPI — ₹1 lakh per transaction, ₹2 lakh per day. "
    "NEFT — no upper limit. IMPS — ₹5 lakh per transaction. "
    "Debit card POS — ₹2 lakh per day (Basic), ₹5 lakh per day (Premium).",

    # --- Account Statement ---
    "You can download your account statement from the app → Accounts → Statement. "
    "Statements are available in PDF format for up to 2 years. You can also request a "
    "physical statement to be mailed to your registered address for ₹50 per copy.",

    # --- Nominee ---
    "To add or update a nominee on your NeoBank account, go to app → Profile → Nominee Details. "
    "You will need the nominee's Aadhaar number and relationship. Nominee can be updated "
    "anytime without visiting a branch. Having a nominee ensures smooth claim settlement.",

    # --- Tax / TDS ---
    "NeoBank deducts TDS at 10% on Fixed Deposit interest exceeding ₹40,000 per financial year "
    "(₹50,000 for senior citizens). You can submit Form 15G/15H through the app to avoid TDS "
    "if your total income is below the taxable limit.",
]


# ============================================================
# RAG ENGINE (unified interface)
# ============================================================

class RAGEngine:
    """
    Production RAG engine supporting:
    - Dual vector store backends (Qdrant / SimpleVectorStore)
    - Hybrid retrieval (BM25 + Vector + RRF fusion)
    - Dynamic document ingestion from files
    - Fallback to hardcoded FAQ if no files found

    VECTOR_DB_MODE env var:
      "memory" (default) → In-memory SimpleVectorStore
      "qdrant"           → Qdrant (Docker or Cloud)
    """

    def __init__(self):
        self._model = _embedding_model  # Shared global model instance
        self._bm25_index = BM25Index()

        # Select vector store backend
        db_mode = os.getenv("VECTOR_DB_MODE", "memory").lower()

        if db_mode == "qdrant" and QDRANT_AVAILABLE:
            try:
                self.vector_store = QdrantVectorStore()
                self._db_mode = "qdrant"
                print("[RAG] Using Qdrant vector database")
            except Exception as e:
                print(f"[RAG] ⚠️ Qdrant connection failed, falling back to memory: {e}")
                self.vector_store = SimpleVectorStore()
                self._db_mode = "memory"
        else:
            self.vector_store = SimpleVectorStore()
            self._db_mode = "memory"
            if db_mode == "qdrant" and not QDRANT_AVAILABLE:
                print("[RAG] ⚠️ VECTOR_DB_MODE=qdrant but qdrant-client not installed. Using memory.")
            print("[RAG] Using in-memory vector store")

    def _embed_text(self, text: str) -> list[float]:
        """Convert text to 384-dim vector using local sentence-transformers model."""
        if not self._model:
            return [0.0] * VECTOR_DIM
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query string. Same as _embed_text for symmetric models like MiniLM."""
        if not self._model:
            return [0.0] * VECTOR_DIM
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def add_documents(self, documents: list[str], id_prefix: str = "doc"):
        """
        Add documents to both vector store and BM25 index.
        Called by ingest.py for dynamically loaded documents.
        """
        if not EMBEDDINGS_AVAILABLE:
            print("[RAG] ⚠️ Skipping document embedding — sentence-transformers not available")
            return

        import time
        start = time.time()

        embeddings = [self._embed_text(doc) for doc in documents]
        ids = [f"{id_prefix}_{i}" for i in range(len(documents))]

        self.vector_store.add(ids=ids, embeddings=embeddings, documents=documents)

        elapsed = round((time.time() - start) * 1000)
        print(f"[RAG] Embedded and stored {len(documents)} documents in {elapsed}ms")

        # Rebuild BM25 index with all documents
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """Rebuild BM25 index from all documents currently in the vector store."""
        if not BM25_AVAILABLE:
            return
        all_docs = self.vector_store.get_all_documents()
        self._bm25_index.build_index(all_docs)

    def initialize(self, external_documents: Optional[list[str]] = None):
        """
        Initialize the RAG engine:
        1. If external_documents provided (from ingest.py), use those
        2. Otherwise, if vector store is empty, embed the FAQ seed data
        3. Build BM25 index from all documents

        Args:
            external_documents: Optional list of document chunks from ingest.py
        """
        if self.vector_store.count() > 0 and external_documents is None:
            print(f"[RAG] Vector store already has {self.vector_store.count()} documents, skipping embedding")
            # Still build BM25 index from existing documents
            self._rebuild_bm25()
            return

        if not EMBEDDINGS_AVAILABLE:
            print("[RAG] ⚠️ Skipping FAQ embedding — sentence-transformers not available")
            return

        # Determine which documents to embed
        if external_documents and len(external_documents) > 0:
            docs_to_embed = external_documents
            id_prefix = "ingested"
            print(f"[RAG] Embedding {len(docs_to_embed)} ingested documents...")
        else:
            docs_to_embed = FAQ_DOCUMENTS
            id_prefix = "faq"
            print(f"[RAG] No external documents provided. Embedding {len(docs_to_embed)} FAQ seed documents...")

        import time
        start = time.time()

        embeddings = [self._embed_text(doc) for doc in docs_to_embed]
        self.vector_store.add(
            ids=[f"{id_prefix}_{i}" for i in range(len(docs_to_embed))],
            embeddings=embeddings,
            documents=docs_to_embed,
        )

        elapsed = round((time.time() - start) * 1000)
        print(f"[RAG] ✅ Embedded {len(docs_to_embed)} documents in {elapsed}ms (local CPU)")

        # Build BM25 index
        self._rebuild_bm25()

    def retrieve(self, query: str, n_results: int = 2) -> list[str]:
        """
        Hybrid retrieval: BM25 keyword search + Vector similarity, merged via RRF.

        Pipeline:
        1. BM25: Get top-10 keyword-matched doc_ids
        2. Vector: Get top-10 semantically similar doc_ids
        3. RRF: Fuse both ranked lists (k=60)
        4. Return top n_results document texts

        Falls back to vector-only if BM25 is unavailable.
        """
        if self.vector_store.count() == 0:
            return []

        query_embedding = self._embed_query(query)

        # Candidate retrieval pool size (retrieve more than needed for fusion)
        pool_size = max(10, n_results * 5)

        # --- Vector search path ---
        vector_results = self.vector_store.query(query_embedding, n_results=pool_size)
        vector_ranked_ids = [doc_id for doc_id, _, _ in vector_results]

        # Build id → text lookup from vector results
        id_to_text = {doc_id: text for doc_id, text, _ in vector_results}

        # --- BM25 search path ---
        if BM25_AVAILABLE and self._bm25_index.bm25 is not None:
            bm25_ranked_ids = self._bm25_index.search(query, n_results=pool_size)

            # Add BM25 results to text lookup (may include docs not in vector top-N)
            all_docs = self.vector_store.get_all_documents()
            all_docs_map = {doc_id: text for doc_id, text in all_docs}
            for doc_id in bm25_ranked_ids:
                if doc_id not in id_to_text:
                    id_to_text[doc_id] = all_docs_map.get(doc_id, "")

            # RRF fusion of both ranked lists
            fused_ids = reciprocal_rank_fusion(
                [vector_ranked_ids, bm25_ranked_ids],
                k=60
            )
        else:
            # Vector-only fallback
            fused_ids = vector_ranked_ids

        # Return top n_results document texts
        results = []
        for doc_id in fused_ids[:n_results]:
            text = id_to_text.get(doc_id, "")
            if text:
                results.append(text)

        return results

    def get_query_embedding(self, query: str) -> list[float]:
        """Expose query embedding for speculative RAG cache similarity checks."""
        return self._embed_query(query)
