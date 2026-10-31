"""A tiny scalar autograd engine.

Each Value wraps one number. Every operation creates a new Value that remembers
its parents and a closure (_backward) that pushes its gradient back to them.
Calling backward() on the final result walks the graph in reverse topological
order and applies the chain rule.
"""
import math


class Value:
    def __init__(self, data, _parents=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0  # d(output)/d(self), filled by backward()
        # _backward(out) pushes out.grad to the parents. It takes out as an argument
        # instead of closing over it, so out -> _backward -> out is not a reference
        # cycle and each graph is freed as soon as it goes out of scope.
        self._backward = lambda out: None
        self._prev = set(_parents)
        self._op = _op  # for graph visualization
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- binary ops -------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward(out):
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward(out):
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward(out):
            # d(x^k)/dx = k * x^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    # Everything else is built from the ops above, so it gets gradients for free.
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):  # 2 + Value
        return self + other

    def __rsub__(self, other):  # 2 - Value
        return other + (-self)

    def __rmul__(self, other):  # 2 * Value
        return self * other

    def __rtruediv__(self, other):  # 2 / Value
        return other * self**-1

    # --- unary ops --------------------------------------------------------

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward(out):
            # d(e^x)/dx = e^x
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward(out):
            # d(tanh x)/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward(out):
            # gradient passes through only where the input was positive
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # --- backprop ---------------------------------------------------------

    def backward(self):
        # Topological order: every node comes after all nodes it depends on.
        # Iterative DFS so deep graphs don't hit Python's recursion limit.
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((p, False) for p in node._prev if p not in visited)

        # Gradients accumulate with +=, which handles a value used several times.
        self.grad = 1.0  # d(self)/d(self)
        for node in reversed(topo):
            node._backward(node)
