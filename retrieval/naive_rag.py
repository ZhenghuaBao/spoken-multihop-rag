"""
Naive RAG retrieval: FAISS + text-embedding-3-small.

Supports both top-1 queries and N-best multi-query strategies.
Reuses DualRAG's core/similarity_search.py patterns.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class NaiveRAG:
    """FAISS-based dense retrieval for spoken multi-hop QA."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        index_path: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=2000)
        self.vectorstore: Optional[FAISS] = None
        self.index_path = index_path

    def index_documents(self, docs: List[str], save_path: Optional[str] = None) -> None:
        """
        Build FAISS index from document strings.

        Args:
            docs: list of document texts
            save_path: optional path to persist the index
        """
        save_path = save_path or self.index_path
        documents = [
            Document(page_content=doc, metadata={"doc_id": i})
            for i, doc in enumerate(docs)
        ]

        print(f"Indexing {len(documents)} documents with {self.embedding_model}...")
        start = time.time()
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        elapsed = time.time() - start
        print(f"Indexing complete in {elapsed:.1f}s")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(save_path)
            print(f"Saved index to {save_path}")

    def load_index(self, path: Optional[str] = None) -> None:
        """Load persisted FAISS index."""
        path = path or self.index_path
        if not path:
            raise ValueError("No index path specified")
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        print(f"Loaded index from {path}")

    def retrieve_top1(
        self,
        query: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Retrieve documents using a single (top-1 ASR) query.

        Returns:
            dict with docs, scores, retrieval_time
        """
        if not self.vectorstore:
            raise RuntimeError(
                "No index loaded. Call index_documents() or load_index() first."
            )

        start = time.time()
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        elapsed = time.time() - start

        return {
            "docs": [doc.page_content for doc, _ in results],
            "scores": [float(score) for _, score in results],
            "retrieval_time": elapsed,
            "query": query,
            "strategy": "top1",
        }

    def retrieve_nbest_union(
        self,
        hypotheses: List[Dict],
        top_k: int = 10,
    ) -> Dict:
        """
        Strategy A: Retrieve separately per hypothesis, union & deduplicate results.
        Rank by best score across hypotheses.

        Args:
            hypotheses: list of {text, score/normalized_score}
            top_k: final number of docs to return
        """
        if not self.vectorstore:
            raise RuntimeError("No index loaded.")

        start = time.time()

        # Retrieve per hypothesis
        all_results = {}  # doc_content -> best_score
        for hyp in hypotheses:
            results = self.vectorstore.similarity_search_with_score(
                hyp["text"], k=top_k
            )
            for doc, score in results:
                key = doc.page_content
                # FAISS L2 distance: lower = better
                if key not in all_results or score < all_results[key]:
                    all_results[key] = score

        # Sort by best score (ascending for L2 distance)
        sorted_docs = sorted(all_results.items(), key=lambda x: x[1])[:top_k]
        elapsed = time.time() - start

        return {
            "docs": [doc for doc, _ in sorted_docs],
            "scores": [float(s) for _, s in sorted_docs],
            "retrieval_time": elapsed,
            "num_hypotheses": len(hypotheses),
            "strategy": "nbest_union",
        }

    def retrieve_nbest_weighted(
        self,
        hypotheses: List[Dict],
        top_k: int = 10,
    ) -> Dict:
        """
        Strategy B: Weighted embedding ensemble.
        Compute weighted average of query embeddings, retrieve with that.

        Args:
            hypotheses: list of {text, normalized_score}
            top_k: number of docs to return
        """
        if not self.vectorstore:
            raise RuntimeError("No index loaded.")

        start = time.time()

        # Embed all hypotheses
        texts = [h["text"] for h in hypotheses]
        weights = [h.get("normalized_score", 1.0 / len(hypotheses)) for h in hypotheses]

        embeddings = self.embeddings.embed_documents(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        weights_np = np.array(weights, dtype=np.float32)

        # Weighted average embedding
        weighted_emb = np.average(embeddings_np, axis=0, weights=weights_np)
        weighted_emb = weighted_emb / np.linalg.norm(weighted_emb)  # normalize

        # Search with the combined embedding
        results = self.vectorstore.similarity_search_with_score_by_vector(
            weighted_emb.tolist(), k=top_k
        )
        elapsed = time.time() - start

        return {
            "docs": [doc.page_content for doc, _ in results],
            "scores": [float(score) for _, score in results],
            "retrieval_time": elapsed,
            "num_hypotheses": len(hypotheses),
            "strategy": "nbest_weighted",
        }

    def retrieve_nbest_concat(
        self,
        hypotheses: List[Dict],
        top_k: int = 10,
    ) -> Dict:
        """
        Strategy C: Concatenate hypotheses into a single enriched query.

        Args:
            hypotheses: list of {text, normalized_score}
            top_k: number of docs to return
        """
        if not self.vectorstore:
            raise RuntimeError("No index loaded.")

        # Build concatenated query (ordered by score)
        sorted_hyps = sorted(hypotheses, key=lambda h: h.get("score", 0), reverse=True)
        concat_query = " | ".join(h["text"] for h in sorted_hyps)

        start = time.time()
        results = self.vectorstore.similarity_search_with_score(concat_query, k=top_k)
        elapsed = time.time() - start

        return {
            "docs": [doc.page_content for doc, _ in results],
            "scores": [float(score) for _, score in results],
            "retrieval_time": elapsed,
            "num_hypotheses": len(hypotheses),
            "strategy": "nbest_concat",
        }


if __name__ == "__main__":
    # Quick test
    rag = NaiveRAG()
    test_docs = [
        "# Paris\n\nParis is the capital of France.",
        "# Berlin\n\nBerlin is the capital of Germany.",
        "# Tokyo\n\nTokyo is the capital of Japan.",
    ]
    rag.index_documents(test_docs)
    result = rag.retrieve_top1("What is the capital of France?", top_k=2)
    print("\nQuery: What is the capital of France?")
    print(f"Top result: {result['docs'][0][:80]}...")
    print(f"Retrieval time: {result['retrieval_time']:.3f}s")
