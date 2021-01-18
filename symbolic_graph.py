# Accumulation path
import string

import torch.autograd as autograd
import torch.nn.functional

from accumulation import plot_policies
from env import make_graph
from torch_model import ResidualBlock

torch.manual_seed(0)

_jacobian_keys = [
    ('ResidualBlock/16', 'output/output.1'),
    ('ResidualBlock/out', 'ResidualBlock/16'),
    ('ResidualBlock/Linear[linear2]/out.2', 'ResidualBlock/out'),
    ('input/input.1', 'ResidualBlock/out'),

    ('ResidualBlock/input', 'ResidualBlock/Linear[linear2]/out.2'),
    ('ResidualBlock/Linear[linear2]/31', 'ResidualBlock/Linear[linear2]/out.2'),
    ('ResidualBlock/Linear[linear2]/weight/30', 'ResidualBlock/Linear[linear2]/31'),

    ('ResidualBlock/Linear[linear1]/out.1', 'ResidualBlock/input'),
    ('ResidualBlock/Linear[linear1]/28', 'ResidualBlock/Linear[linear1]/out.1'),
    ('ResidualBlock/Linear[linear1]/weight/27', 'ResidualBlock/Linear[linear1]/28'),
    ('input/input.1', 'ResidualBlock/Linear[linear1]/out.1')

]
graph_keys = ['input/input.1', 'output/output.1', 'ResidualBlock/Linear[linear1]/weight/27',
              'ResidualBlock/Linear[linear1]/28', 'ResidualBlock/Linear[linear1]/out.1', 'ResidualBlock/input',
              'ResidualBlock/Linear[linear2]/weight/30', 'ResidualBlock/Linear[linear2]/31',
              'ResidualBlock/Linear[linear2]/out.2', 'ResidualBlock/out', 'ResidualBlock/16']
_symbolic_graph = {}
for c, key in zip(string.ascii_lowercase, graph_keys):
    _symbolic_graph[key] = c

_symbolic_jacobian = {}
for key in _jacobian_keys:
    parent, child = key
    symbolic_key = (_symbolic_graph[parent], _symbolic_graph[child])
    _symbolic_jacobian[key] = symbolic_key


def get_torch_jacobian():
    num_channels = 1
    model = ResidualBlock(num_channels)
    x = torch.ones(size=(1,))
    out = model(x)
    out.backward()
    grads = {}
    for idx, v in enumerate(model.parameters()):
        grads[f"dy/dw:w{idx}"] = v.grad.clone()
    model.zero_grad()
    return model, grads


# 5 operations and 4 variables (2 x w, 2 in/out)
def get_elemental_jacobian(model):
    x = torch.ones(size=(1,))
    linear_1 = model.linear1
    linear_2 = model.linear2
    _jacobian = {}

    ### BLOCK 1
    dx, dw = autograd.functional.jacobian(torch.nn.functional.linear, inputs=(x, linear_1.weight))
    key = _symbolic_jacobian[('ResidualBlock/Linear[linear1]/28', 'ResidualBlock/Linear[linear1]/out.1')]
    _jacobian[key] = dw
    key = _symbolic_jacobian[('input/input.1', 'ResidualBlock/Linear[linear1]/out.1')]
    _jacobian[key] = dx
    key = _symbolic_jacobian[('ResidualBlock/Linear[linear1]/weight/27', 'ResidualBlock/Linear[linear1]/28')]
    _jacobian[key] = torch.ones_like(dw)

    h = torch.nn.functional.linear(x, weight=linear_1.weight)
    key = _symbolic_jacobian[('ResidualBlock/Linear[linear1]/out.1', 'ResidualBlock/input')]
    _jacobian[key] = autograd.functional.jacobian(lambda x: x, h)
    h = h
    # ('a,e') = 0
    ## BLOCK 2
    dx, dw = autograd.functional.jacobian(torch.nn.functional.linear, inputs=(h, linear_2.weight))
    key = _symbolic_jacobian[('ResidualBlock/Linear[linear2]/31', 'ResidualBlock/Linear[linear2]/out.2')]
    _jacobian[key] = dw
    key = _symbolic_jacobian[('ResidualBlock/input', 'ResidualBlock/Linear[linear2]/out.2')]
    _jacobian[key] = dx
    key = _symbolic_jacobian[('ResidualBlock/Linear[linear2]/weight/30', 'ResidualBlock/Linear[linear2]/31')]
    _jacobian[key] = torch.ones_like(dw)

    h = torch.nn.functional.linear(h, weight=linear_2.weight)

    ## BLOCK 3
    dx, dw = autograd.functional.jacobian(lambda x, h: x + h, inputs=(x, h))
    key = _symbolic_jacobian[('input/input.1', 'ResidualBlock/out')]
    _jacobian[key] = dx
    key = _symbolic_jacobian[('ResidualBlock/Linear[linear2]/out.2', 'ResidualBlock/out')]
    _jacobian[key] = dw
    output = h + x

    ## BLOCK 4
    key = _symbolic_jacobian[('ResidualBlock/out', 'ResidualBlock/16')]
    _jacobian[key] = autograd.functional.jacobian(lambda x: x, output)
    y = output

    ## OUT
    key = _symbolic_jacobian[('ResidualBlock/16', 'output/output.1')]
    _jacobian[key] = autograd.functional.jacobian(lambda x: x, y)

    return _jacobian


policy = [
    'ResidualBlock/Linear[linear2]/out.2',
    'ResidualBlock/Linear[linear2]/31',
    'ResidualBlock/16',
    'ResidualBlock/input',
    'ResidualBlock/Linear[linear1]/28',
    'ResidualBlock/Linear[linear1]/out.1',
    'ResidualBlock/out'
]
reverse_mode = [
    'ResidualBlock/16',
    'ResidualBlock/out',
    'ResidualBlock/Linear[linear2]/out.2',
    'ResidualBlock/Linear[linear2]/31',
    'ResidualBlock/input',
    'ResidualBlock/Linear[linear1]/out.1',
    'ResidualBlock/Linear[linear1]/28',
]


def renamed_graph():
    graph = make_graph("graph")
    new_graph = {}
    for node in graph._graph.values():
        node.name = _symbolic_graph[node.name]
        new_inputs = set()
        for input in node.input:
            new_inputs.add(_symbolic_graph[input])
        node.input = new_inputs
        new_graph[node.name] = node
    graph._edges = {}
    graph._graph = new_graph
    graph.compute_edges()
    return graph


def get_policies():
    oja_policy = [_symbolic_graph[p] for p in policy]
    reverse_policy = [_symbolic_graph[p] for p in policy]
    return oja_policy, reverse_policy

def save_graph():
    graph = renamed_graph()
    model, torch_jacobian = get_torch_jacobian()
    j = get_elemental_jacobian(model)
    for edge_key, edge in graph._edges.items():
        edge.jacobian = j[edge_key]
        print(edge)

    oja_policy, reverse_policy = get_policies()
    with open("renamed_graph.pkl", "wb") as f:
        pickle.dump((graph, (oja_policy, reverse_policy)), f)

