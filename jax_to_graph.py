import haiku as hk
import jax
import jax.numpy as jnp
import tree
from haiku._src.dot import to_graph, _max_depth, Graph

# TODO
"""
add label to this graph and make it compatibile with env
try ve on this graph
piece it together with jacobian chaining
"""


class Resnet(hk.Module):

    def __init__(self, h_dim=1):
        super(Resnet, self).__init__(name="resnet")
        self.fc = hk.Sequential([hk.Linear(h_dim),
                                 jax.nn.relu,
                                 hk.Linear(h_dim)
                                 ]
                                )

    def __call__(self, x):
        h = self.fc(x)
        # return h + x
        return jax.nn.relu(x + h)


def resnet(x):
    return Resnet()(x)


class Node:

    def __init__(self, name, op, id):
        self.name: str = name
        self.op: str = op
        self.id: int = id
        self.input: set = set()
        self.output: set = set()

    def __repr__(self):
        return f"op:{self.name}\tin:{self.input}\tout:{self.output}"


class Edge:
    input: str
    output: str
    count: int = 0
    jacobian = 0

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.input}->{self.output} {float(self.jacobian):.4f}"

    @property
    def label(self):
        return self.__repr__()


# How to get the gradient to propagate backward?
h = jnp.ones((1,))
key_gen = hk.PRNGSequence(42)
resnet = hk.without_apply_rng(hk.transform(resnet))
resnet_params = resnet.init(next(key_gen), h)

graph, args, out = to_graph(resnet.apply)(resnet_params, h)


def format_path(path, outputs, prefix='output'):
    if isinstance(outputs, tuple):
        out = f'{prefix}[{path[0]}]'
        if len(path) > 1:
            out += ': ' + '/'.join(map(str, path[1:]))
    else:
        out = prefix
        if path:
            out += ': ' + '/'.join(map(str, path))
    return out


def format_array(array):
    return f"{array.shape}[{array.dtype}]"


def parse_graph(graph, args, outputs):
    nodes = {}
    edges = []
    depth = _max_depth(graph)

    # op_out = {}
    output_name = {id(v): format_path(p, outputs) for p, v in tree.flatten_with_path(outputs)}
    input_name = {id(v): format_path(p, args, 'input') for p, v in tree.flatten_with_path(args)}
    op_out = set()
    intermediates = {}
    unique_keys = set()

    def _parse_graph(graph: Graph, root, depth):
        for node in graph.nodes:
            _node = Node(name=f"{node.title}:{graph.title}", op=node.title, id=id(node.id))
            assert _node.id not in nodes.keys()
            [_node.output.add(id(o)) for o in node.outputs]
            nodes[_node.id] = _node
            for o in node.outputs:
                op_out.add(id(o))

        for subgraph in graph.subgraphs:
            _parse_graph(subgraph, graph, depth - 1)

        for a, b in graph.edges:
            if id(a) not in input_name.keys() and id(a) not in op_out:
                label = graph.title + "\t" + format_array(a)
                intermediates[id(a)] = label
            a, b = map(id, (a, b))
            edges.append((a, b))
            unique_keys.add(a)
            unique_keys.add(b)

    _parse_graph(graph, None, depth)
    for key, node in nodes.items():
        for (a, b) in edges:
            if b == node.id:
                node.input.add(a)
    return nodes, edges


nodes, edges = parse_graph(graph, args, out)
import graphviz

print(nodes, edges)

out = hk.experimental.to_dot(resnet.apply)(resnet_params, h)
graphviz.Source(out).render(view=True)
print("")
