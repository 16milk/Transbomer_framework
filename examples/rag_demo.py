"""Run a deterministic local RAG pipeline with MiniTensor reranking.

Run from the repository root:
    python examples/rag_demo.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import minitensor as mt


class Reranker(mt.Module):
    """Tiny MLP that learns to score retrieval and lexical-overlap features."""

    def __init__(self) -> None:
        super().__init__()
        self.network = mt.Sequential(mt.Linear(2, 8), mt.ReLU(), mt.Linear(8, 1))

    def forward(self, features: mt.Tensor) -> mt.Tensor:
        return self.network(features)


def lexical_overlap(query: str, text: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
    text_words = set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
    return len(query_words & text_words) / max(len(query_words), 1)


def make_features(
    queries: list[str],
    chunks: list[mt.DocumentChunk],
    encoder: mt.HashingEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[float] = []
    for query in queries:
        query_vector = encoder.encode_one(query)[0]
        document_vectors = encoder.encode([chunk.text for chunk in chunks])
        similarities = mt.cosine_similarity(query_vector, document_vectors).data
        target_source = next(
            source
            for keyword, source in (
                ("自动微分", "autograd"),
                ("Transformer", "transformer"),
                ("序列化", "serialization"),
            )
            if keyword.lower() in query.lower()
        )
        for chunk, similarity in zip(chunks, similarities):
            rows.append([float(similarity), lexical_overlap(query, chunk.text)])
            labels.append(float(Path(chunk.source).stem == target_source))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64).reshape(-1, 1)


def main() -> None:
    mt.set_seed(19)
    data_dir = Path(__file__).parent / "data" / "rag_docs"
    chunks = mt.load_markdown(data_dir.glob("*.md"), max_chars=220)
    encoder = mt.HashingEncoder(dimension=48)

    queries = [
        "自动微分如何计算梯度",
        "Transformer 支持什么 attention 组件",
        "序列化如何保存模型参数",
    ]
    features, labels = make_features(queries, chunks, encoder)
    reranker = Reranker()
    optimizer = mt.Adam(reranker.parameters(), lr=0.05)
    feature_tensor = mt.tensor(features)
    label_tensor = mt.tensor(labels)
    for _ in range(180):
        optimizer.zero_grad()
        loss = mt.mse_loss(reranker(feature_tensor), label_tensor)
        loss.backward()
        optimizer.step()
    reranker.eval()

    print(f"chunks={len(chunks)}, reranker_loss={loss.item():.6f}")
    for query in queries:
        candidates = mt.retrieve(query, chunks, encoder, top_k=3)
        candidate_features = np.asarray(
            [
                [result.score, lexical_overlap(query, result.chunk.text)]
                for result in candidates
            ],
            dtype=np.float64,
        )
        rerank_scores = reranker(mt.tensor(candidate_features)).data.reshape(-1)
        order = np.argsort(-rerank_scores, kind="stable")
        print(f"\nquery: {query}")
        for index in order:
            result = candidates[index]
            print(
                f"- {result.chunk.chunk_id} retrieval={result.score:.4f} "
                f"rerank={rerank_scores[index]:.4f}: {result.chunk.text[:100]}"
            )


if __name__ == "__main__":
    main()
