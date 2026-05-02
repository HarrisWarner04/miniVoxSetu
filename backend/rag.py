"""
miniVoxSetu — RAG (Retrieval-Augmented Generation) Engine

WHY THIS FILE EXISTS:
LLMs like Gemini are trained on public internet data — they know nothing about
YOUR specific business (NeoBank in our case). RAG solves this by:
1. Converting your domain knowledge (FAQ) into vector embeddings
2. Storing those embeddings in a vector database (ChromaDB in production)
3. On each user query, finding the most relevant knowledge chunks
4. Injecting those chunks into the LLM prompt

This is EXACTLY how production AI agents handle domain knowledge without
fine-tuning. VoxSetu and similar systems use this pattern at scale.

NOTE ON VECTOR STORE:
We use a lightweight NumPy-based vector store here because ChromaDB's C++ extension
(chroma-hnswlib) doesn't have pre-built wheels for Python 3.13 on Windows yet.
The interface is IDENTICAL to what ChromaDB provides — when ChromaDB adds 3.13
support, you can swap in chromadb.Client() with zero changes to the rest of the code.
To use ChromaDB instead, just: pip install chromadb (requires Python ≤3.12 or C++ build tools)
"""

import numpy as np
import google.generativeai as genai


# ============================================================
# SIMPLE VECTOR STORE (mirrors ChromaDB's interface)
# ============================================================

class SimpleVectorStore:
    """
    WHY THIS CLASS EXISTS:
    In production, you'd use ChromaDB, Pinecone, Weaviate, or pgvector.
    This is a minimal implementation that teaches the EXACT same concepts:
    - Storing documents with their vector embeddings
    - Querying by cosine similarity
    - Returning the top-N most relevant documents

    The math is simple: cosine_similarity(A, B) = dot(A, B) / (|A| * |B|)
    Similar texts have vectors pointing in the same direction → high cosine similarity.

    ChromaDB equivalent:
        collection = chroma_client.get_or_create_collection("neobank_faq")
        collection.add(ids=[...], embeddings=[...], documents=[...])
        results = collection.query(query_embeddings=[...], n_results=2)
    """

    def __init__(self):
        self.documents = []      # List of document strings
        self.embeddings = []     # List of embedding vectors (numpy arrays)
        self.ids = []            # Document IDs

    def add(self, ids, embeddings, documents):
        """
        WHY: Store documents alongside their embeddings. This is the "indexing"
        step — converting text to vectors and saving them for later search.
        In ChromaDB: collection.add(ids=ids, embeddings=embeddings, documents=documents)
        """
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.embeddings.extend([np.array(e) for e in embeddings])

    def query(self, query_embedding, n_results=2):
        """
        WHY: Find the most similar documents to the query using cosine similarity.
        This is the core of vector search — instead of matching KEYWORDS (like SQL LIKE),
        we match MEANING. "How do I open an account?" matches a document about account
        opening even if the exact words differ.

        In ChromaDB: collection.query(query_embeddings=[query_emb], n_results=n)

        HOW COSINE SIMILARITY WORKS:
        Each embedding is a vector in high-dimensional space (768 dimensions for
        text-embedding-004). Cosine similarity measures the angle between two vectors:
        - cos(θ) = 1.0 → identical direction → identical meaning
        - cos(θ) = 0.0 → perpendicular → unrelated
        - cos(θ) = -1.0 → opposite → opposite meaning
        """
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding)

        # WHY: We compute cosine similarity against ALL stored documents.
        # In production with millions of docs, this brute-force approach is too slow
        # and you'd use approximate nearest neighbor (ANN) algorithms like HNSW
        # (which is exactly what ChromaDB uses internally via hnswlib).
        similarities = []
        for i, doc_vec in enumerate(self.embeddings):
            # Cosine similarity formula: dot(A,B) / (||A|| * ||B||)
            cos_sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            similarities.append((i, cos_sim))

        # WHY: Sort by similarity (highest first) and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:n_results]]

        return [self.documents[i] for i in top_indices]

    def count(self):
        return len(self.documents)


# ============================================================
# FAQ KNOWLEDGE BASE
# ============================================================

# WHY: These are our "knowledge base" documents. In production, these would
# come from a CMS, database, or document store. We hardcode them here for
# learning purposes. Each string is a "chunk" — a self-contained piece of
# information that can be retrieved independently.
FAQ_DOCUMENTS = [
    "NeoBank offers three account types: Basic (no minimum balance, free), "
    "Premium ($10/month, higher ATM limits), and Business ($25/month, multi-user access and invoicing).",

    "To open a NeoBank account, you need a valid government-issued ID, proof of address less than "
    "3 months old, and you must be at least 18 years old. The process takes under 5 minutes online.",

    "NeoBank's interest rate on savings accounts is 4.5% APY for Premium members and 2.1% APY "
    "for Basic members. Interest is compounded daily and paid out monthly.",

    "NeoBank supports instant transfers to any bank in the US via ACH (free, 1-2 business days) "
    "or wire transfer ($15 fee, same-day). International transfers are available via SWIFT for $30.",

    "If your NeoBank debit card is lost or stolen, immediately freeze it in the app under "
    "Settings → Cards → Freeze Card. A replacement card ships within 2 business days at no charge.",

    "NeoBank's customer support is available 24/7 via in-app chat. Phone support is available "
    "Monday to Friday 8 AM to 8 PM EST at 1-800-NEO-BANK. Premium members get priority queue.",

    "NeoBank charges no overdraft fees. If your balance goes negative, transactions are simply "
    "declined. You can opt into Overdraft Protection which links to your savings account.",

    "NeoBank's mobile app supports biometric login (Face ID, fingerprint), push notifications "
    "for all transactions, spending insights with AI categorization, and bill splitting with friends.",
]


class RAGEngine:
    """
    WHY THIS CLASS ENCAPSULATES ALL RAG LOGIC:
    Separation of concerns — the main server shouldn't know HOW retrieval works,
    just that it can call retrieve(query) and get relevant text back. This mirrors
    production architecture where the RAG system is often a separate microservice.
    """

    def __init__(self, api_key: str):
        """
        WHY: We configure the Gemini SDK here for embedding. We use a SEPARATE
        model for embeddings vs. chat — embedding models are optimized for
        converting text to vectors, not for generating text. This is a key
        architectural concept: different models for different tasks.
        """
        genai.configure(api_key=api_key)

        # WHY: We use our SimpleVectorStore here. In production, you'd use:
        #   self.chroma_client = chromadb.Client()
        #   self.collection = self.chroma_client.get_or_create_collection("neobank_faq")
        # The interface is the same — add() to index, query() to search.
        self.vector_store = SimpleVectorStore()

    def _embed_text(self, text: str) -> list[float]:
        """
        WHY: This converts human-readable text into a vector (list of numbers)
        that captures its MEANING. Similar texts will have similar vectors.
        This is the foundation of semantic search — we're not matching keywords,
        we're matching MEANING. "How do I open an account?" will match the
        document about account opening even though the exact words differ.
        """
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def _embed_query(self, text: str) -> list[float]:
        """
        WHY: Query embeddings use a different task_type than document embeddings.
        This tells the model "this is a search query" vs "this is a document to index".
        The model optimizes the embedding differently — query embeddings are tuned
        to be close to relevant documents in vector space. This asymmetry improves
        retrieval quality significantly.
        """
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def initialize(self):
        """
        WHY: We embed ALL FAQ documents at startup and store them in the vector store.
        This is the "indexing" phase of RAG. In production, this would happen
        offline (in a batch job) and the index would be persisted to disk.
        Embedding is expensive (API calls), so you only want to do it once.
        """
        # WHY: We check if docs already exist to avoid re-embedding on hot reload.
        # Embedding API calls cost time and (in production) money.
        if self.vector_store.count() > 0:
            print(f"[INFO] Vector store already has {self.vector_store.count()} documents, skipping embedding")
            return

        print("[INFO] Embedding FAQ documents into vector store...")

        embeddings = []
        for doc in FAQ_DOCUMENTS:
            embedding = self._embed_text(doc)
            embeddings.append(embedding)

        # WHY: We add all documents in a single batch call. This is more efficient
        # than adding one at a time — fewer round trips to the database.
        self.vector_store.add(
            ids=[f"faq_{i}" for i in range(len(FAQ_DOCUMENTS))],
            embeddings=embeddings,
            documents=FAQ_DOCUMENTS,
        )
        print(f"[OK] Embedded {len(FAQ_DOCUMENTS)} FAQ documents")

    def retrieve(self, query: str, n_results: int = 2) -> list[str]:
        """
        WHY: This is the "retrieval" step that runs on EVERY user query.
        We convert the user's question into a vector, then find the closest
        document vectors in our store. "Closest" means "most semantically similar".
        We return the top N results to inject into the LLM prompt.

        WHY n_results=2: We retrieve 2 chunks by default because:
        1. Too few → might miss relevant context
        2. Too many → wastes context window tokens and can confuse the LLM
        In production, you'd tune this based on chunk size and model context limits.
        """
        if self.vector_store.count() == 0:
            return []

        query_embedding = self._embed_query(query)
        return self.vector_store.query(query_embedding, n_results=n_results)
