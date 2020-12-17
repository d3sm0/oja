import collections
from typing import NamedTuple, Dict, Any, Tuple

import numpy as np

PATH = {"x1": ((), ("v1",)),
        "x2": ((), ("v1",)),
        "v1": (("x1", "x2"), ("v2", "y1")),
        "v2": (("v1",), ("v3",)),
        "v3": (("v2",), ("y2",)),
        "y2": (("v3",), ()),
        "y1": (("v1", "v3"), ()),
        }


class Node(NamedTuple):
    key: str
    parents: tuple
    children: tuple


class Graph:
    def __init__(self):
        self._graph = {}
        self._connectivity = {}
        self._vertexes = {}
        self._idx = 0
        self._cost = collections.defaultdict(lambda: 0)

    def add_node(self, node: Node):
        self._graph[node.key] = node
        self._vertexes[node.key] = self._idx
        self._idx += 1  # fast retrivial of nodes
        for s in node.parents:
            self._connectivity[(node.key, s)] = 1
        for t in node.children:
            self._connectivity[(node.key, t)] = 1

    def get_pa(self):
        pa = []
        for node in self._graph.values():
            if len(node.children) == 0 or len(node.parents) == 0:
                continue
            pa.append(node.key)
        return pa

    def add_edge(self, src: str, target: str, value: int = 1):
        self._connectivity[(src, target)] = value

    def update(self, key: str):
        node = self._graph.pop(key)
        for parent in node.parents:
            for child in node.children:
                self.add_edge(parent, child)
                self.add_edge(parent, key, value=0)
                self.add_edge(key, child, value=0)
        return node

    def get_connectivity(self) -> np.ndarray:
        n = len(self._vertexes)
        out = np.zeros((n, n))
        # Meh?
        for edge, value in self._connectivity.items():
            r, c = edge
            r = self._vertexes[r]
            c = self._vertexes[c]
            out[r, c] = value
        return out


def make_graph() -> Graph:
    graph = Graph()
    for k in PATH.keys():
        node = Node(k, *PATH[k])
        graph.add_node(node)
    return graph


class Env:
    def __init__(self):
        self.graph = make_graph()
        self.t = 0
        self.history = []

    def step(self, action: str) -> Tuple[np.ndarray, int, bool, Dict[str, Any]]:
        m0 = self.graph.get_connectivity()
        node = self.graph.update(action)
        m1 = self.graph.get_connectivity()
        self.history.append(node)

        added = np.clip(m1 - m0, 0, 1)
        deleted = np.abs(np.clip(m1 - m0, -1, 0))

        stats = {"adds": added.sum(), "del": deleted.sum()}
        total_cost = -stats["adds"]
        pa = self.graph.get_pa()
        done = not bool(len(pa))
        info = {
            "pa": pa,
            **stats
        }
        self.t += 1
        return m1, total_cost, done, info

    def reset(self) -> Tuple[np.ndarray, int, bool, Dict[str, Any]]:
        self.t = 0
        self.graph = make_graph()

        m0 = self.graph.get_connectivity()
        info = {
            "pa": self.graph.get_pa()
        }

        return m0, 0, False, info

    def render(self):
        return


def main():
    env = Env()
    m0, _, done, info = env.reset()
    total_reward = 0
    while not done:
        action = np.random.choice(info["pa"])
        m1, reward, done, info = env.step(action)
        total_reward += reward
    print(total_reward)


if __name__ == '__main__':
    main()
