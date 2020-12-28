# from __future__ import annotations
import collections
import copy
import itertools
import os
import pickle
from typing import NamedTuple, Dict, Any, Tuple, Union, List

import gym
import numpy as np

from utils import operation_count
from utils import plot_computational_graph

PATH = {"x1": (),
        "x2": (),
        "x3": (),
        "v1": ("x1", "x2", "x3"),
        "v2": ("v1",),
        "y1": ("v2", "x1",),
        "y2": ("v2",),
        "y3": ("v2", "v1"),
        }


class GameState(NamedTuple):
    state: np.ndarray
    # reward: float
    done: bool
    info: Dict[str, Union[List, Any]]


class Node(NamedTuple):
    name: str
    op: str = ""
    idx: int = 0
    input: set = set()


class Edge(NamedTuple):
    input: str
    child: str
    label: str = ""


class Graph:
    def __init__(self):
        self._graph = {}
        self._edges = []
        self._total_nodes = 0
        self._cost = collections.defaultdict(lambda: 0)

    def add_node(self, node: Node):
        self._graph[node.name] = node
        self._total_nodes += 1
        for s in node.input:
            self._edges.append((s, node.name))

    def get_pa(self):
        pa = []
        for node in self._graph.values():
            if "IO" in node.op or len(node.input) == 0:
                continue
            pa.append(node.name)
        return pa

    def update(self, key: str):
        node = self._graph.pop(key)
        in_edge = set()
        out_edge = set()
        for edge in self._edges:
            if node.name == edge[0]:
                out_edge.add(edge[1])
            elif node.name == edge[1]:
                in_edge.add(edge[0])
            else:
                continue
        for child, parent in itertools.product(*[out_edge, in_edge]):
            while True:
                try:
                    self._edges.remove((key, child))
                except ValueError:
                    break
            while True:
                try:
                    self._edges.remove((parent, key))
                except ValueError:
                    break
            self._edges.append((parent, child))
        for edge in self._edges:
            assert key != edge[0] and key != edge[1]

        return node

    def get_connectivity(self) -> np.ndarray:
        n = self._total_nodes
        out = np.zeros((n, n))
        for edge in self._edges:
            r, c = edge
            r = self._graph[r].idx
            c = self._graph[c].idx
            out[r, c] += 1
        return out


def save_graph():
    nodes = []
    for idx, (node_name, edge) in enumerate(PATH.items()):
        op = "IO" if "y" in node_name else "no_op"
        nodes.append(Node(name=node_name, idx=idx, op=op, input=edge))
    with open("assets/basic_graph.pkl", "wb") as f:
        pickle.dump((nodes, {}), f)


save_graph()


def make_graph(fname) -> Graph:
    graph = Graph()
    with open(f"assets/{fname}.pkl", "rb") as f:
        nodes, _ = pickle.load(f)
    for idx, node in enumerate(nodes):
        graph.add_node(Node(name=node.name, idx=idx, input=node.input, op=node.op))

    return graph


class Env(gym.Env):
    def __init__(self, fname="graph"):
        self.fname = fname
        self.graph = make_graph(fname)
        print("bwd op count:", operation_count(self.graph.get_connectivity()))
        self.action_space = gym.spaces.Discrete(len(self.graph._graph.keys()))
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
        self.graph = make_graph(self.fname)

        m0 = self.graph.get_connectivity()
        info = {
            "pa": self.graph.get_pa()
        }

        return m0, 0, False, info

    def render(self, mode="human"):
        plot_computational_graph(self.graph._edges, os.path.join("assets", f"{self.fname}_{self.t}"))

    def get_state(self) -> GameState:
        state = self.graph.get_connectivity()
        info = {
            "pa": self.graph.get_pa(),
            "graph": copy.deepcopy(self.graph)
        }
        done = not bool(len(info["pa"]))
        game_state = GameState(state, done, info)
        return game_state

    def set_state(self, state: GameState) -> GameState:
        graph = copy.deepcopy(state.info["graph"])
        self.graph = graph
        return state


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
