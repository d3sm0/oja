from typing import Tuple

import numpy as np


def plot_mcts(root, fname=""):
    """
    Plot the MCTS, pdf file is saved in the current directory.
    """
    try:
        from graphviz import Digraph
    except ModuleNotFoundError:
        print("Please install graphviz to get the MCTS plot.")
        return None

    graph = Digraph(comment="MCTS", engine="dot")
    graph.attr("graph", rankdir="LR", splines="true", overlap="false")
    id = 0

    def traverse(node, action, parent_id, best):
        nonlocal id
        node_id = id
        graph.node(
            str(node_id),
            label=node.__repr__(),
            color="orange" if best else "black",
        )
        id += 1
        if parent_id is not None:
            graph.edge(str(parent_id), str(node_id), constraint="false")

        if len(node.children) != 0:
            best_visit_count = max(
                [child.explore_count for child in node.children.values()]
            )
        else:
            best_visit_count = False
        for action, child in node.children.items():
            if child.explore_count != 0:
                traverse(
                    child,
                    action,
                    node_id,
                    True
                    if best_visit_count and child.explore_count == best_visit_count
                    else False,
                )

    traverse(root, None, None, True)
    graph.node(str(0), color="red")
    # print(graph.source)
    graph.render(fname, view=False, cleanup=True, format="png")
    return graph


def plot_computational_graph(graph, fname=""):
    try:
        from graphviz import Digraph
    except ModuleNotFoundError:
        print("Please install graphviz to get the MCTS plot.")
        return None

    digraph = Digraph(comment="graph", engine="dot")
    digraph.attr("graph", rankdir="lR", size="8,5")
    for edges in graph:
        digraph.edge(*edges)
    digraph.render(fname, view=False, format="png", cleanup=True)


def back_substitution(matrix: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, int]:
    n = b.size
    x = np.zeros_like(b)
    op_count = 0
    if matrix[n - 1, n - 1] == 0:
        raise ValueError

    for i in range(n - 1, 0, -1):
        x[i] = matrix[i, i] / b[i]
        for j in range(i - 1, 0, -1):
            matrix[i, i] += matrix[j, i] * x[i]
            op_count += 1
    return x, op_count


def operation_count(jacobian):
    import scipy.linalg as la
    out = la.lu(jacobian)
    #  l is the linearization of the extended jacobian
    p, l, u = out
    n, m = l.shape
    vector = np.ones(m)
    bwd_mode, op_count = back_substitution(np.eye(n), vector.T)
    return op_count


if __name__ == '__main__':
    operation_count(jacobian=np.eye(10))
