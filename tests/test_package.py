import minitensor


def test_package_imports() -> None:
    assert minitensor.__version__ == "0.1.0"
    assert minitensor.__all__ == ["MiniTensorError", "__version__"]

