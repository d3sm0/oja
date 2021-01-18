import pickle

import torch
import torch.utils.tensorboard as tb

from env import Node, Edge


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def linear(in_channels, out_channels, stride=1):
    return torch.nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
    # return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.linear1 = linear(num_channels, num_channels)
        # self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.linear2 = linear(num_channels, num_channels)
        # self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.linear1(x)
        # out = self.bn1(out)
        out = out
        out = self.linear2(out)
        # out = self.bn2(out)
        out = x + out
        out = out
        return out


def _test_model():
    model = ResidualBlock(1)
    # model = torch.nn.Sequential(*[torch.nn.Linear(1, 2), torch.nn.ReLU(inplace=True)])
    from profile import count_ops_torch
    model_input = torch.zeros((1,))
    import copy
    count_ops_torch(copy.deepcopy(model), input_size=model_input.shape)

    with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
        try:
            trace = torch.jit.trace(model, model_input)
            # graph = trace.graph
            torch._C._jit_pass_inline(trace.graph)
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e
    list_of_nodes = tb._pytorch_graph.parse(trace.graph, trace, model_input)
    nodes = []
    edges = []
    for node in list_of_nodes:
        if "aten" in node.op or "GetAttr" in node.op or "IO Node" in node.op:
            simple_node = Node(
                name=node.name,
                op=node.op,
                input=set(node.input)
            )
            if "aten" in node.op:
                for input in node.input:
                    edges.append(Edge(input=input, child=node.name, label=node.op))
            nodes.append(simple_node)

    node_names = [node.name for node in nodes]
    new_nodes = []
    for node in nodes:
        import copy
        inputs = copy.deepcopy(node.input)
        for input in node.input:
            if input not in node_names:
                inputs.remove(input)
        new_nodes.append(Node(name=node.name, op=node.op, input=inputs))

    for node in new_nodes:
        assert node.name in node_names
        for input in node.input:
            assert input in node_names

    clean_edges = []
    for edge in edges:
        if edge.input in node_names:
            clean_edges.append(edge)
    # plot_computational_graph(clean_edges)
    with open("assets/graph.pkl", "wb") as f:
        pickle.dump((new_nodes, clean_edges), f)
    # writer = tb.SummaryWriter("logs/")
    # writer.add_graph(model, input_to_model=, verbose=True)
    # writer.close()


if __name__ == '__main__':
    _test_model()
