import numpy as np

import minitensor as mt


def test_softmax_and_log_softmax_are_stable_for_extreme_logits():
    logits = mt.tensor([[1000.0, 999.0, -1000.0], [-1000.0, -999.0, 1000.0]])
    probabilities = mt.softmax(logits)
    log_probabilities = mt.log_softmax(logits)

    assert np.isfinite(probabilities.data).all()
    assert np.isfinite(log_probabilities.data).all()
    np.testing.assert_allclose(probabilities.data.sum(axis=-1), 1.0)
    expected_log_probabilities = logits.data - logits.data.max(axis=-1, keepdims=True)
    expected_log_probabilities -= np.log(
        np.exp(expected_log_probabilities).sum(axis=-1, keepdims=True)
    )
    np.testing.assert_allclose(log_probabilities.data, expected_log_probabilities)


def test_embedding_accumulates_repeated_index_gradients():
    embedding = mt.Embedding(5, 3)
    output = embedding(np.array([[1, 2, 1]]))
    output.sum().backward()

    np.testing.assert_allclose(embedding.weight.grad.data[1], np.ones(3) * 2)
    np.testing.assert_allclose(embedding.weight.grad.data[2], np.ones(3))
    np.testing.assert_allclose(embedding.weight.grad.data[[0, 3, 4]], 0)


def test_layer_norm_matches_numpy_reference():
    values = np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 7.0]])
    layer = mt.LayerNorm(3, eps=1e-5)
    actual = layer(mt.tensor(values)).data
    centered = values - values.mean(axis=-1, keepdims=True)
    expected = centered / np.sqrt(values.var(axis=-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_dropout_changes_training_output_but_is_identity_in_eval():
    np.random.seed(4)
    dropout = mt.Dropout(0.5)
    inputs = mt.tensor(np.ones((1000,)))
    training_output = dropout(inputs).data
    assert set(np.unique(training_output)).issubset({0.0, 2.0})
    assert 0.9 < training_output.mean() < 1.1

    dropout.eval()
    np.testing.assert_array_equal(dropout(inputs).data, inputs.data)


def test_attention_mask_and_forward_backward():
    np.random.seed(2)
    attention = mt.MultiHeadAttention(4, 2)
    inputs = mt.tensor(np.random.randn(2, 3, 4))
    mask = np.zeros((3, 3), dtype=bool)
    mask[:, -1] = True
    outputs = attention(inputs, mask=mask)

    assert outputs.shape == inputs.shape
    outputs.mean().backward()
    assert all(parameter.grad is not None for parameter in attention.parameters())


def test_transformer_block_supports_batched_matmul_and_gradients():
    np.random.seed(8)
    block = mt.TransformerBlock(8, 2, feed_forward_dim=16, dropout=0.0)
    inputs = mt.tensor(np.random.randn(3, 4, 8))
    outputs = block(inputs)

    assert outputs.shape == (3, 4, 8)
    outputs.mean().backward()
    assert all(parameter.grad is not None for parameter in block.parameters())
