import numpy as np

import minitensor as mt


def test_split_and_hashing_encoder_are_deterministic():
    text = "First paragraph.\n\nSecond paragraph with enough words."
    chunks = mt.split_text(text, source="notes.md", max_chars=12)
    assert [chunk.chunk_id for chunk in chunks] == [f"notes:{index}" for index in range(5)]

    encoder = mt.HashingEncoder(dimension=16)
    first = encoder.encode(["same text", "other text"]).data
    second = mt.HashingEncoder(dimension=16).encode(["same text", "other text"]).data
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)


def test_retrieve_returns_stable_top_match():
    chunks = [
        mt.DocumentChunk("a:0", "a.md", "autograd computes gradients with backward"),
        mt.DocumentChunk("b:0", "b.md", "attention uses query key and value"),
    ]
    results = mt.retrieve("how does autograd compute gradients", chunks, mt.HashingEncoder(32))

    assert results[0].chunk.chunk_id == "a:0"
    assert results[0].score >= results[1].score
