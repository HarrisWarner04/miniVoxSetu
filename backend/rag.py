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
