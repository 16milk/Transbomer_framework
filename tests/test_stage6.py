import numpy as np
import pytest

import minitensor as mt


def test_state_dict_round_trip_preserves_model_output(tmp_path):
    mt.set_seed(7)
    model = mt.Sequential(mt.Linear(2, 3), mt.ReLU(), mt.Linear(3, 1))
    inputs = mt.tensor([[1.0, -2.0], [0.5, 3.0]])
    expected = model(inputs).data.copy()
    path = tmp_path / "model.npz"

    mt.save_state_dict(model, path)
    restored = mt.Sequential(mt.Linear(2, 3), mt.ReLU(), mt.Linear(3, 1))
    metadata = mt.load_state_dict(restored, path)

    np.testing.assert_allclose(restored(inputs).data, expected)
    assert metadata["module"] == "Sequential"
    assert metadata["parameters"]["0.weight"]["shape"] == [2, 3]


def test_load_state_dict_reports_shape_dtype_and_key_errors():
    model = mt.Linear(2, 1)
    with pytest.raises(mt.StateDictError, match="shape"):
        model.load_state_dict(
            {"weight": np.zeros((3, 1), dtype=model.weight.dtype), "bias": model.bias.data}
        )
    with pytest.raises(mt.StateDictError, match="missing keys"):
        model.load_state_dict({"weight": model.weight.data})
    with pytest.raises(mt.StateDictError, match="dtype"):
        model.load_state_dict(
            {"weight": model.weight.data.astype(np.float32), "bias": model.bias.data}
        )


def test_seed_and_tools_are_reproducible_and_report_useful_stats():
    mt.set_seed(11)
    first = mt.Linear(3, 2).weight.data.copy()
    mt.set_seed(11)
    second = mt.Linear(3, 2).weight.data.copy()
    np.testing.assert_array_equal(first, second)

    model = mt.Sequential(mt.Linear(3, 2), mt.ReLU())
    assert mt.count_parameters(model) == 8
    summary = mt.module_summary(model)
    assert "total_parameters: 8" in summary
    result = mt.benchmark(lambda: model(mt.ones((4, 3))), iterations=2)
    assert result.iterations == 2
    assert result.mean_seconds >= result.min_seconds >= 0


def test_graph_stats_reports_active_graph_counts():
    graph = mt.trace(lambda x: (x * 2).relu(), np.ones((2, 2), dtype=np.float32))
    stats = mt.graph_stats(graph)
    assert stats == {"nodes": 2, "inputs": 1, "outputs": 1}
