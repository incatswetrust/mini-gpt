"""Draw the computation graph of a Value with graphviz.

Needs the Python package (`pip install graphviz`) and, to render images,
the system binary (`brew install graphviz`).
"""
from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()
    stack = [root]
    while stack:
        v = stack.pop()
        if v in nodes:
            continue
        nodes.add(v)
        for p in v._prev:
            edges.add((p, v))
            stack.append(p)
    return nodes, edges


def draw_dot(root, rankdir="LR"):
    """Value boxes show label | data | grad; small circles are the ops."""
    dot = Digraph(format="svg", graph_attr={"rankdir": rankdir})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(uid, label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}", shape="record")
        if n._op:
            dot.node(uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for a, b in edges:
        dot.edge(str(id(a)), str(id(b)) + b._op)
    return dot


if __name__ == "__main__":
    from engine import Value

    # Single neuron: o = tanh(x1*w1 + x2*w2 + b)
    x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
    w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")
    x1w1 = x1 * w1; x1w1.label = "x1*w1"
    x2w2 = x2 * w2; x2w2.label = "x2*w2"
    n = x1w1 + x2w2 + b; n.label = "n"
    o = n.tanh(); o.label = "o"
    o.backward()
    draw_dot(o).render("neuron_graph", cleanup=True)
    print("saved neuron_graph.svg")
