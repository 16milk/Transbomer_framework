"""Static graph tracing, execution, and small graph optimizations.

The graph IR is intentionally small.  Values carry only compile-time metadata
and Nodes reference eager MiniTensor kernels by name, which keeps the static
runtime consistent with the eager runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .tensor import Tensor, add, div, exp, log, matmul, max, mean, mul, pow, relu, sigmoid, sub, sum


@dataclass
class Value:
    """An SSA-like graph value."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    producer: Node | None = None
    const: Tensor | None = None
    alias: Value | None = None

    @property
    def is_constant(self) -> bool:
        return self.const is not None or (self.alias is not None and self.alias.is_constant)

    def resolve(self) -> Value:
        value = self
        while value.alias is not None:
            value = value.alias
        return value

    def __repr__(self) -> str:
        suffix = " constant" if self.is_constant else ""
        return f"{self.name}: {self.shape} {self.dtype.name}{suffix}"


@dataclass
class Node:
    """A single operation in a Graph."""

    op: str
    inputs: tuple[Value, ...]
    output: Value
    attrs: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def __repr__(self) -> str:
        arguments = ", ".join(value.resolve().name for value in self.inputs)
        return f"{self.output.name} = {self.op}({arguments})"


def _as_constant(value: Any, name: str) -> Value:
    tensor_value = value if isinstance(value, Tensor) else Tensor(value)
    return Value(name, tensor_value.shape, tensor_value.dtype, const=tensor_value)


class Graph:
    """A statically traced computation graph."""

    def __init__(self) -> None:
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []
        self.nodes: list[Node] = []
        self._value_index = 0

    @property
    def active_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.active]

    def _new_name(self, prefix: str) -> str:
        name = f"{prefix}{self._value_index}"
        self._value_index += 1
        return name

    def input(self, shape: tuple[int, ...], dtype: Any, name: str | None = None) -> Value:
        value = Value(name or self._new_name("input"), tuple(shape), np.dtype(dtype))
        self.inputs.append(value)
        return value

    def constant(self, value: Any, name: str | None = None) -> Value:
        return _as_constant(value, name or self._new_name("const"))

    def node(
        self,
        op: str,
        inputs: Iterable[Value | Any],
        *,
        attrs: dict[str, Any] | None = None,
        output: Tensor | None = None,
    ) -> Value:
        normalized = tuple(
            item if isinstance(item, Value) else self.constant(item) for item in inputs
        )
        concrete_inputs = tuple(
            item.const if item.const is not None else Tensor(np.zeros(item.shape, dtype=item.dtype))
            for item in normalized
        )
        result = _execute(op, concrete_inputs, attrs or {}) if output is None else output
        value = Value(
            self._new_name(op),
            result.shape,
            result.dtype,
        )
        created = Node(op, normalized, value, attrs or {})
        value.producer = created
        self.nodes.append(created)
        return value

    def topological_sort(self) -> list[Node]:
        """Return active nodes in dependency order."""
        ordered: list[Node] = []
        visited: set[int] = set()

        def visit(value: Value) -> None:
            value = value.resolve()
            if value.producer is None:
                return
            node = value.producer
            if not node.active or id(node) in visited:
                return
            for input_value in node.inputs:
                visit(input_value)
            visited.add(id(node))
            ordered.append(node)

        for output in self.outputs:
            visit(output)
        return ordered

    def run(self, *inputs: Any) -> Tensor | tuple[Tensor, ...]:
        """Execute the graph with runtime input tensors."""
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Expected {len(self.inputs)} graph inputs, received {len(inputs)}.")
        environment: dict[int, Tensor] = {
            id(value): (item if isinstance(item, Tensor) else Tensor(item))
            for value, item in zip(self.inputs, inputs)
        }
        for node in self.topological_sort():
            arguments = tuple(_lookup(environment, value) for value in node.inputs)
            environment[id(node.output.resolve())] = _execute(node.op, arguments, node.attrs)
        results = tuple(_lookup(environment, output) for output in self.outputs)
        return results[0] if len(results) == 1 else results

    def optimize(self) -> Graph:
        """Apply constant folding, CSE, and dead-code elimination in place."""
        replacements: dict[int, Value] = {}
        cse: dict[tuple[Any, ...], Value] = {}
        for node in self.nodes:
            if not node.active:
                continue
            node.inputs = tuple(_replace(value, replacements) for value in node.inputs)
            if all(value.resolve().is_constant for value in node.inputs):
                constants = tuple(value.resolve().const for value in node.inputs)
                assert all(item is not None for item in constants)
                folded = _execute(node.op, constants, node.attrs)
                node.output.const = folded
                node.active = False
                continue
            key = (
                node.op,
                tuple(_cse_key(value) for value in node.inputs),
                tuple(sorted((key, repr(value)) for key, value in node.attrs.items())),
            )
            previous = cse.get(key)
            if previous is not None:
                node.active = False
                replacements[id(node.output)] = previous
            else:
                cse[key] = node.output
        self.outputs = [_replace(value, replacements) for value in self.outputs]
        self._eliminate_dead_nodes()
        return self

    def _eliminate_dead_nodes(self) -> None:
        needed = {id(node) for node in self.topological_sort()}
        for node in self.nodes:
            node.active = node.active and id(node) in needed

    def __str__(self) -> str:
        lines = ["Graph("]
        lines.append("  inputs: " + ", ".join(repr(value) for value in self.inputs))
        for node in self.topological_sort():
            attributes = f" {node.attrs}" if node.attrs else ""
            lines.append(f"  {node}{attributes}")
        lines.append("  outputs: " + ", ".join(value.resolve().name for value in self.outputs))
        lines.append(")")
        return "\n".join(lines)


class _Tracer:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def value(self, value: Any) -> Value:
        if isinstance(value, Value):
            return value
        if isinstance(value, Proxy):
            return value.value
        return self.graph.constant(value)

    def unary(self, op: str, value: Any) -> Value:
        return self.graph.node(op, (self.value(value),))

    def binary(self, op: str, left: Any, right: Any) -> Value:
        return self.graph.node(op, (self.value(left), self.value(right)))

    def reduce(self, op: str, value: Any, axis: Any = None, keepdims: bool = False) -> Value:
        return self.graph.node(op, (self.value(value),), attrs={"axis": axis, "keepdims": keepdims})


class Proxy:
    """Tensor-like object used only while tracing."""

    def __init__(self, tracer: _Tracer, value: Value) -> None:
        self._tracer, self.value = tracer, value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.value.dtype

    @property
    def T(self) -> Proxy:
        return self.transpose()

    def _wrap(self, value: Value) -> Proxy:
        return Proxy(self._tracer, value)

    def reshape(self, *shape: Any) -> Proxy:
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return self._wrap(
            self._tracer.graph.node("reshape", (self.value,), attrs={"shape": tuple(target)})
        )

    def transpose(self, *axes: Any) -> Proxy:
        target = None if not axes else (axes[0] if len(axes) == 1 else axes)
        if isinstance(target, list):
            target = tuple(target)
        return self._wrap(
            self._tracer.graph.node("transpose", (self.value,), attrs={"axes": target})
        )

    def sum(self, axis: Any = None, keepdims: bool = False) -> Proxy:
        return self._wrap(self._tracer.reduce("sum", self.value, axis, keepdims))

    def mean(self, axis: Any = None, keepdims: bool = False) -> Proxy:
        return self._wrap(self._tracer.reduce("mean", self.value, axis, keepdims))

    def max(self, axis: Any = None, keepdims: bool = False) -> Proxy:
        return self._wrap(self._tracer.reduce("max", self.value, axis, keepdims))

    def exp(self) -> Proxy:
        return self._wrap(self._tracer.unary("exp", self.value))

    def log(self) -> Proxy:
        return self._wrap(self._tracer.unary("log", self.value))

    def relu(self) -> Proxy:
        return self._wrap(self._tracer.unary("relu", self.value))

    def sigmoid(self) -> Proxy:
        return self._wrap(self._tracer.unary("sigmoid", self.value))

    def __getitem__(self, index: Any) -> Proxy:
        return self._wrap(self._tracer.graph.node("getitem", (self.value,), attrs={"index": index}))

    def __add__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("add", self.value, other))

    __radd__ = __add__

    def __sub__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("sub", self.value, other))

    def __rsub__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("sub", other, self.value))

    def __mul__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("mul", self.value, other))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("div", self.value, other))

    def __rtruediv__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("div", other, self.value))

    def __pow__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("pow", self.value, other))

    def __rpow__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("pow", other, self.value))

    def __matmul__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("matmul", self.value, other))

    def __rmatmul__(self, other: Any) -> Proxy:
        return self._wrap(self._tracer.binary("matmul", other, self.value))

    def __neg__(self) -> Proxy:
        return self._wrap(self._tracer.unary("neg", self.value))


def trace(function: Callable[..., Any], *example_inputs: Any) -> Graph:
    """Trace ``function`` once using example inputs and return a Graph."""
    graph = Graph()
    tracer = _Tracer(graph)
    proxies = [
        Proxy(tracer, graph.input(Tensor(item).shape, Tensor(item).dtype, name=f"input{i}"))
        for i, item in enumerate(example_inputs)
    ]
    result = function(*proxies)
    if isinstance(result, Proxy):
        graph.outputs = [result.value]
    elif isinstance(result, (tuple, list)) and all(isinstance(item, Proxy) for item in result):
        graph.outputs = [item.value for item in result]
    else:
        raise TypeError("Traced function must return a Proxy or a tuple/list of Proxy values.")
    return graph


def _replace(value: Value, replacements: dict[int, Value]) -> Value:
    current = value.resolve()
    while id(current) in replacements:
        current = replacements[id(current)].resolve()
    return current


def _cse_key(value: Value) -> tuple[Any, ...]:
    resolved = value.resolve()
    if resolved.const is not None:
        data = resolved.const.data
        return ("const", data.dtype.str, data.shape, data.tobytes())
    return ("value", id(resolved))


def _lookup(environment: dict[int, Tensor], value: Value) -> Tensor:
    resolved = value.resolve()
    if resolved.const is not None:
        return resolved.const
    return environment[id(resolved)]


def _execute(op: str, inputs: tuple[Tensor, ...], attrs: dict[str, Any]) -> Tensor:
    if op == "reshape":
        return inputs[0].reshape(attrs["shape"])
    if op == "transpose":
        return (
            inputs[0].transpose()
            if attrs["axes"] is None
            else inputs[0].transpose(attrs["axes"])
        )
    if op == "getitem":
        return inputs[0][attrs["index"]]
    functions = {
        "add": add, "sub": sub, "mul": mul, "div": div, "pow": pow, "matmul": matmul,
        "exp": exp, "log": log, "relu": relu, "sigmoid": sigmoid,
        "sum": sum, "mean": mean, "max": max,
    }
    if op == "neg":
        return -inputs[0]
    if op in {"sum", "mean", "max"}:
        return functions[op](inputs[0], axis=attrs["axis"], keepdims=attrs["keepdims"])
    if op not in functions:
        raise ValueError(f"Unsupported graph operation: {op}")
    return functions[op](*inputs)
