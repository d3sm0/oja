import copy
import pickle

from utils import plot_computational_graph


def load_graph():
    with open("renamed_graph.pkl", "rb") as f:
        graph, (oja, reverse) = pickle.load(f)
    return graph, oja, reverse


def plot_policy(graph, policy, policy_name):
    for idx, action in enumerate(policy):
        old_node = graph.update(action)
        plot_computational_graph(graph._edges, f"policies/{policy_name}_{idx}", view=True)

    for edge in graph._edges.values():
        print(edge)


def plot_policies():
    graph, oja, reverse = load_graph()
    plot_computational_graph(graph._edges, "policies/graph.png", view=True)
    plot_policy(copy.deepcopy(graph), oja, policy_name="oja")
    plot_policy(graph, reverse, policy_name="reverse")


if __name__ == '__main__':
    plot_policies()
