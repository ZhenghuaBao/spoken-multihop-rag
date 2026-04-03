"""
HippoRAG wrapper for spoken multi-hop QA.

Handles Windows vllm compatibility (vllm is Linux-only) by mocking
required modules before importing hipporag.
"""

import sys
import time
import types
from pathlib import Path
from typing import Dict, List


def _patch_vllm_for_windows():
    """Mock vllm and Unix-only modules so hipporag can import on Windows."""
    # Mock 'resource' (Unix-only)
    if "resource" not in sys.modules:
        try:
            import resource  # noqa
        except ModuleNotFoundError:
            mock = types.ModuleType("resource")
            mock.RLIMIT_NOFILE = 7
            mock.getrlimit = lambda *a: (1024, 65536)
            mock.setrlimit = lambda *a, **kw: None
            sys.modules["resource"] = mock

    # Mock vllm (Linux-only GPU inference engine)
    try:
        import vllm  # noqa
    except (ImportError, ModuleNotFoundError, OSError):
        vllm_mock = types.ModuleType("vllm")
        vllm_mock.__path__ = []
        vllm_mock.__file__ = ""

        sub_mods = [
            "vllm",
            "vllm._C",
            "vllm.engine",
            "vllm.engine.arg_utils",
            "vllm.config",
            "vllm.model_executor",
            "vllm.model_executor.parameter",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.quantization",
            "vllm.distributed",
            "vllm.distributed.communication_op",
            "vllm.distributed.parallel_state",
            "vllm.distributed.kv_transfer",
            "vllm.distributed.kv_transfer.kv_transfer_agent",
            "vllm.distributed.kv_transfer.kv_connector",
            "vllm.distributed.kv_transfer.kv_connector.factory",
            "vllm.distributed.kv_transfer.kv_connector.base",
            "vllm.sequence",
            "vllm.inputs",
            "vllm.inputs.registry",
            "vllm.transformers_utils",
            "vllm.transformers_utils.tokenizer",
            "vllm.utils",
            "vllm.platforms",
            "vllm.platforms.cuda",
        ]
        for mod_name in sub_mods:
            if mod_name not in sys.modules:
                m = types.ModuleType(mod_name)
                m.__path__ = []
                m.__file__ = ""
                sys.modules[mod_name] = m

        class _MockSamplingParams:
            def __init__(self, **kw):
                pass

        class _MockLLM:
            def __init__(self, **kw):
                pass

        sys.modules["vllm"].SamplingParams = _MockSamplingParams
        sys.modules["vllm"].LLM = _MockLLM


# Patch BEFORE any hipporag import
_patch_vllm_for_windows()

HIPPORAG_AVAILABLE = False
try:
    from hipporag import HippoRAG as _HippoRAG

    HIPPORAG_AVAILABLE = True
except Exception as e:
    print(f"WARNING: Failed to import hipporag: {type(e).__name__}: {e}")
    _HippoRAG = None


class SpokenHippoRAG:
    """HippoRAG wrapper with N-best query support for spoken input."""

    def __init__(
        self,
        save_dir: str = "hipporag_outputs",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        num_to_retrieve: int = 20,
        num_for_generation: int = 10,
    ):
        self.save_dir = save_dir
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.num_to_retrieve = num_to_retrieve
        self.num_for_generation = num_for_generation
        self.hipporag = None

    def initialize(self, docs: List[str], force_reindex: bool = False) -> None:
        """Initialize HippoRAG with document corpus."""
        if not HIPPORAG_AVAILABLE:
            raise ImportError(
                "hipporag not available. Install with: pip install hipporag"
            )

        save_path = Path(self.save_dir)
        need_index = force_reindex or not save_path.exists()

        print("Initializing HippoRAG...")
        print(f"  LLM: {self.llm_model}")
        print(f"  Embedding: {self.embedding_model}")
        print(f"  Save dir: {self.save_dir}")

        self.hipporag = _HippoRAG(
            save_dir=self.save_dir,
            llm_model_name=self.llm_model,
            embedding_model_name=self.embedding_model,
        )

        if need_index:
            print(f"Indexing {len(docs)} documents...")
            start = time.time()
            self.hipporag.index(docs=docs)
            print(f"Indexing done in {time.time() - start:.1f}s")
        else:
            print(f"Using existing index from {save_path}")

    def retrieve_top1(self, query: str) -> Dict:
        """Retrieve using a single (top-1 ASR) query."""
        if not self.hipporag:
            raise RuntimeError("Not initialized. Call initialize() first.")

        start = time.time()
        solutions = self.hipporag.retrieve(
            queries=[query], num_to_retrieve=self.num_to_retrieve
        )
        elapsed = time.time() - start

        docs = []
        if solutions and hasattr(solutions[0], "docs"):
            docs = (solutions[0].docs or [])[: self.num_for_generation]

        return {
            "docs": docs,
            "retrieval_time": elapsed,
            "query": query,
            "strategy": "top1",
        }

    def retrieve_nbest_union(self, hypotheses: List[Dict]) -> Dict:
        """
        Strategy A: Retrieve per hypothesis, union results.

        HippoRAG extracts entities from queries for KG traversal.
        Multiple hypotheses → more entity variants → better KG coverage.
        """
        if not self.hipporag:
            raise RuntimeError("Not initialized.")

        start = time.time()

        all_docs = []
        seen = set()

        for hyp in hypotheses:
            solutions = self.hipporag.retrieve(
                queries=[hyp["text"]], num_to_retrieve=self.num_to_retrieve
            )
            if solutions and hasattr(solutions[0], "docs"):
                for doc in solutions[0].docs or []:
                    if doc not in seen:
                        all_docs.append(doc)
                        seen.add(doc)

        elapsed = time.time() - start

        docs = all_docs[: self.num_for_generation]

        return {
            "docs": docs,
            "retrieval_time": elapsed,
            "num_hypotheses": len(hypotheses),
            "total_unique_docs": len(all_docs),
            "strategy": "nbest_union",
        }

    def retrieve_nbest_concat(self, hypotheses: List[Dict]) -> Dict:
        """
        Strategy C: Concatenate hypotheses into one query.

        For HippoRAG, this means the NER step sees all entity variants
        in a single pass, which may improve entity extraction.
        """
        if not self.hipporag:
            raise RuntimeError("Not initialized.")

        sorted_hyps = sorted(hypotheses, key=lambda h: h.get("score", 0), reverse=True)
        concat_query = " | ".join(h["text"] for h in sorted_hyps)

        start = time.time()
        solutions = self.hipporag.retrieve(
            queries=[concat_query], num_to_retrieve=self.num_to_retrieve
        )
        elapsed = time.time() - start

        docs = []
        if solutions and hasattr(solutions[0], "docs"):
            docs = (solutions[0].docs or [])[: self.num_for_generation]

        return {
            "docs": docs,
            "retrieval_time": elapsed,
            "num_hypotheses": len(hypotheses),
            "strategy": "nbest_concat",
        }


if __name__ == "__main__":
    print(f"HippoRAG available: {HIPPORAG_AVAILABLE}")
