import numpy as np

import minitensor as mt


def test_trace_runs_same_model_as_eager_and_infers_metadata():
    model = mt.Sequential(mt.Linear(2, 3), mt.ReLU(), mt.Linear(3, 1))
    values = np.array([[1.0, -2.0], [0.5, 3.0]])
    eager = model(mt.tensor(values))

    graph = mt.trace(model, values)
    actual = graph.run(values)

    assert graph.inputs[0].shape == values.shape
    assert graph.inputs[0].dtype == values.dtype
    assert actual.shape == eager.shape
    assert actual.dtype == eager.dtype
    np.testing.assert_allclose(actual.data, eager.data)


def test_graph_text_and_topological_order_are_debuggable():
    graph = mt.trace(lambda x: (x * 2 + 1).relu(), np.ones((2, 3), dtype=np.float32))

    assert [node.op for node in graph.topological_sort()] == ["mul", "add", "relu"]
    text = str(graph)
    assert "input0" in text
    assert "relu" in text
    assert "(2, 3)" in text


def test_constant_folding_and_dead_code_elimination_reduce_nodes():
    def function(x):
        constant_branch = (mt.tensor([2.0]) + mt.tensor([3.0])) * 4
        useful = x + constant_branch
        _unused = x.exp()
        return useful

    graph = mt.trace(function, np.array([1.0]))
    before = len(graph.active_nodes)
    expected = graph.run(np.array([1.0]))
    graph.optimize()
    after = len(graph.active_nodes)

    assert after < before
    np.testing.assert_allclose(graph.run(np.array([1.0])).data, expected.data)
    assert "exp" not in str(graph)


def test_common_subexpression_elimination_preserves_result():
    graph = mt.trace(lambda x: (x * 2) + (x * 2), np.array([1.0, 2.0]))
    before = len(graph.active_nodes)
    expected = graph.run(np.array([3.0, 4.0]))

    graph.optimize()

    assert len(graph.active_nodes) < before
    np.testing.assert_allclose(graph.run(np.array([3.0, 4.0])).data, expected.data)

