"""
miniVoxSetu — Automated Evaluation Harness (Phase 6)

Runs offline benchmarks for RAG retrieval, PII redaction, and latency.
Generates Markdown + JSON reports.

Usage:
  cd backend
  python eval_harness.py              # Run all tests
  python eval_harness.py --rag-only   # RAG tests only
  python eval_harness.py --pii-only   # PII tests only
"""

import sys
import time
import json
import datetime
import os
import argparse
from pathlib import Path

# Ensure we can import project modules
sys.path.insert(0, os.path.dirname(__file__))

from rag import RAGEngine, BM25_AVAILABLE
from pii import redact_pii
from ingest import DocumentIngester


# ============================================================
# RAG EVALUATION
# ============================================================

# (query, expected_substring_in_top_result)
RAG_TEST_CASES = [
    ("How to open an account?", "Aadhaar card"),
    ("What are FD interest rates?", "7.1%"),
    ("Lost card help", "Block Card"),
    ("What is the UPI transaction limit?", "1 lakh"),
    ("How to file a complaint?", "ticket number"),
    ("What are the loan interest rates?", "10.5%"),
    ("How to check EMI details?", "Active Loans"),
    ("NEFT transfer fee", "NEFT is free"),
    ("Senior citizen FD rate", "0.5%"),
    ("How to do KYC?", "Video KYC"),
    ("International card charges", "1.5%"),
    ("TDS on FD", "Form 15G"),
    ("Customer support phone number", "1800-NEO-BANK"),
    ("Cheque book request", "Order Cheque Book"),
    ("Nominee update", "Nominee Details"),
]

# Keyword-heavy queries that BM25 should help with
BM25_TEST_CASES = [
    ("NEFT IMPS RTGS", "NEFT"),
    ("TDS 15G 15H", "15G"),
    ("IFSC code", "NEFT"),  # IFSC mentioned in transfer context
    ("RuPay Visa Platinum", "RuPay"),
    ("overdraft FD protection", "Overdraft"),
]


def evaluate_rag(engine: RAGEngine) -> dict:
    """Run RAG retrieval precision tests."""
    print("\n" + "=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    results = {"precision_at_1": 0, "precision_at_3": 0, "details": [], "latencies_ms": []}

    all_cases = RAG_TEST_CASES + BM25_TEST_CASES
    hits_at_1 = 0
    hits_at_3 = 0

    for query, expected in all_cases:
        start = time.time()
        top_results = engine.retrieve(query, n_results=3)
        latency_ms = round((time.time() - start) * 1000, 1)
        results["latencies_ms"].append(latency_ms)

        hit_at_1 = any(expected.lower() in r.lower() for r in top_results[:1])
        hit_at_3 = any(expected.lower() in r.lower() for r in top_results[:3])

        if hit_at_1:
            hits_at_1 += 1
        if hit_at_3:
            hits_at_3 += 1

        status = "✅" if hit_at_1 else ("⚠️ @3" if hit_at_3 else "❌")
        print(f"  {status} '{query[:40]}' → expected '{expected}' [{latency_ms}ms]")

        results["details"].append({
            "query": query,
            "expected": expected,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "latency_ms": latency_ms,
            "top_result_preview": top_results[0][:80] if top_results else "(empty)",
        })

    total = len(all_cases)
    results["precision_at_1"] = round(hits_at_1 / total, 4)
    results["precision_at_3"] = round(hits_at_3 / total, 4)
    results["avg_latency_ms"] = round(sum(results["latencies_ms"]) / len(results["latencies_ms"]), 1)
    results["total_queries"] = total
    results["bm25_available"] = BM25_AVAILABLE

    print(f"\n  Precision@1: {results['precision_at_1']:.1%} ({hits_at_1}/{total})")
    print(f"  Precision@3: {results['precision_at_3']:.1%} ({hits_at_3}/{total})")
    print(f"  Avg Latency: {results['avg_latency_ms']}ms")
    print(f"  BM25 Active: {BM25_AVAILABLE}")

    return results


# ============================================================
# PII EVALUATION
# ============================================================

PII_TEST_CASES = [
    # (input_text, expected_token, should_redact)
    ("My card is 4111 1111 1111 1111", "[CARD_NO]", True),
    ("Card number 5500000000000004", "[CARD_NO]", True),
    ("Aadhaar 1234 5678 9012", "[AADHAAR]", True),
    ("Aadhaar number is 123456789012", "[AADHAAR]", True),
    ("My PAN is ABCDE1234F", "[PAN]", True),
    ("PAN number ZZZZZ9999Z", "[PAN]", True),
    ("Call me at 9876543210", "[PHONE]", True),
    ("Phone +91 98765 43210", "[PHONE]", True),
    ("Email test@example.com", "[EMAIL]", True),
    ("IFSC code SBIN0001234", "[IFSC]", True),
    ("Account 12345678901234", "[ACCOUNT_NO]", True),
    # False positive checks (should NOT redact)
    ("The interest rate is 7.1% per annum", "[CARD_NO]", False),
    ("Call our helpline at 1800", "[PHONE]", False),
    ("Visit neobank.in for more", "[EMAIL]", False),
]


def evaluate_pii() -> dict:
    """Run PII redaction accuracy tests."""
    print("\n" + "=" * 60)
    print("PII REDACTION EVALUATION")
    print("=" * 60)

    results = {"accuracy": 0, "details": [], "true_positives": 0, "false_negatives": 0, "false_positives": 0}
    correct = 0

    for text, token, should_redact in PII_TEST_CASES:
        redacted, count = redact_pii(text)
        was_redacted = token in redacted

        if should_redact:
            if was_redacted:
                correct += 1
                results["true_positives"] += 1
                print(f"  ✅ Redacted: '{text[:40]}' → {token}")
            else:
                results["false_negatives"] += 1
                print(f"  ❌ MISSED: '{text[:40]}' (expected {token})")
        else:
            if not was_redacted:
                correct += 1
                print(f"  ✅ Preserved: '{text[:40]}'")
            else:
                results["false_positives"] += 1
                print(f"  ⚠️ FALSE POSITIVE: '{text[:40]}' wrongly redacted")

        results["details"].append({
            "input": text, "token": token, "should_redact": should_redact,
            "was_redacted": was_redacted, "correct": (should_redact == was_redacted),
        })

    results["accuracy"] = round(correct / len(PII_TEST_CASES), 4)
    results["total_cases"] = len(PII_TEST_CASES)

    print(f"\n  Accuracy: {results['accuracy']:.1%} ({correct}/{len(PII_TEST_CASES)})")
    print(f"  True Positives: {results['true_positives']} | False Negatives: {results['false_negatives']} | False Positives: {results['false_positives']}")

    return results


# ============================================================
# LATENCY BENCHMARKS
# ============================================================

def benchmark_latency(engine: RAGEngine) -> dict:
    """Benchmark embedding and retrieval latency."""
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARKS")
    print("=" * 60)

    results = {}

    # Embedding latency
    test_text = "What are the interest rates for fixed deposits?"
    times = []
    for _ in range(10):
        start = time.time()
        engine._embed_query(test_text)
        times.append((time.time() - start) * 1000)
    results["embedding_avg_ms"] = round(sum(times) / len(times), 1)
    results["embedding_p95_ms"] = round(sorted(times)[int(len(times) * 0.95)], 1)
    print(f"  Embedding: avg={results['embedding_avg_ms']}ms, p95={results['embedding_p95_ms']}ms")

    # Retrieval latency (includes embedding + search + fusion)
    times = []
    for _ in range(10):
        start = time.time()
        engine.retrieve(test_text, n_results=2)
        times.append((time.time() - start) * 1000)
    results["retrieval_avg_ms"] = round(sum(times) / len(times), 1)
    results["retrieval_p95_ms"] = round(sorted(times)[int(len(times) * 0.95)], 1)
    print(f"  Retrieval: avg={results['retrieval_avg_ms']}ms, p95={results['retrieval_p95_ms']}ms")

    # PII redaction latency
    pii_text = "My card 4111111111111111 and Aadhaar 123456789012 and PAN ABCDE1234F"
    times = []
    for _ in range(100):
        start = time.time()
        redact_pii(pii_text)
        times.append((time.time() - start) * 1000)
    results["pii_avg_ms"] = round(sum(times) / len(times), 2)
    print(f"  PII Redaction: avg={results['pii_avg_ms']}ms (100 runs)")

    return results


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(rag_results: dict, pii_results: dict, latency_results: dict):
    """Generate Markdown and JSON evaluation reports."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(__file__).parent / "reports"
    report_dir.mkdir(exist_ok=True)

    # JSON report
    all_results = {
        "timestamp": timestamp,
        "rag": rag_results,
        "pii": pii_results,
        "latency": latency_results,
    }
    json_path = report_dir / f"eval_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Markdown report
    md_path = report_dir / f"eval_report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# miniVoxSetu — Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%B %d, %Y %H:%M:%S')}\n\n")

        f.write("## RAG Retrieval\n\n")
        f.write(f"| Metric | Value |\n|:---|:---|\n")
        f.write(f"| Precision@1 | {rag_results['precision_at_1']:.1%} |\n")
        f.write(f"| Precision@3 | {rag_results['precision_at_3']:.1%} |\n")
        f.write(f"| Avg Latency | {rag_results['avg_latency_ms']}ms |\n")
        f.write(f"| BM25 Active | {rag_results['bm25_available']} |\n")
        f.write(f"| Total Queries | {rag_results['total_queries']} |\n\n")

        f.write("## PII Redaction\n\n")
        f.write(f"| Metric | Value |\n|:---|:---|\n")
        f.write(f"| Accuracy | {pii_results['accuracy']:.1%} |\n")
        f.write(f"| True Positives | {pii_results['true_positives']} |\n")
        f.write(f"| False Negatives | {pii_results['false_negatives']} |\n")
        f.write(f"| False Positives | {pii_results['false_positives']} |\n\n")

        f.write("## Latency Benchmarks\n\n")
        f.write(f"| Component | Avg | P95 |\n|:---|:---|:---|\n")
        f.write(f"| Embedding | {latency_results['embedding_avg_ms']}ms | {latency_results['embedding_p95_ms']}ms |\n")
        f.write(f"| Retrieval | {latency_results['retrieval_avg_ms']}ms | {latency_results['retrieval_p95_ms']}ms |\n")
        f.write(f"| PII Redaction | {latency_results['pii_avg_ms']}ms | — |\n")

    print(f"\n📄 Reports saved:")
    print(f"  Markdown: {md_path}")
    print(f"  JSON: {json_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="miniVoxSetu Evaluation Harness")
    parser.add_argument("--rag-only", action="store_true", help="Run RAG tests only")
    parser.add_argument("--pii-only", action="store_true", help="Run PII tests only")
    args = parser.parse_args()

    print("=" * 60)
    print("miniVoxSetu — Evaluation Harness")
    print("=" * 60)

    # Initialize RAG engine with document ingestion
    engine = RAGEngine()
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge")
    ingester = DocumentIngester()
    chunks = ingester.ingest_directory(knowledge_dir)
    engine.initialize(external_documents=chunks if chunks else None)
    print(f"[EVAL] RAG engine ready: {engine.vector_store.count()} documents")

    rag_results = {}
    pii_results = {}
    latency_results = {}

    if not args.pii_only:
        rag_results = evaluate_rag(engine)
        latency_results = benchmark_latency(engine)

    if not args.rag_only:
        pii_results = evaluate_pii()

    if rag_results or pii_results:
        generate_report(
            rag_results or {},
            pii_results or {},
            latency_results or {},
        )

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
