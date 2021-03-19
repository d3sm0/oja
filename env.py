# from __future__ import annotations
import collections
# TODO add terminal cost that promotes sparsity of the full Jacobian (e.g. non-zero entries)
# TODO add non-zero entries
# TODO add shape reward for contracting matrices
import copy
import os
import pickle
from typing import NamedTuple, Dict, Any, Tuple, Union, List

import gym
import numpy as np
import torch

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


class Node:
    name: str = ""
    op: str = ""
    idx: int = 0
    input: set = set()

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.name}:{self.input}"


class Edge:
    input: str
    child: str
    count: int = 0
    jacobian = 0

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.input}->{self.child} {float(self.jacobian):.4f}"
    @property
    def label(self):
        return self.__repr__()


class Graph:
    def __init__(self):
        self._graph = {}
        self._edges = {}
        self._total_nodes = 0
        self._cost = {}

    def __iter__(self):
        for name, node in self._graph.items():
            yield name, node

    def compute_edges(self):
        for node_name, node in self._graph.items():
            for parent in node.input:
                edge = Edge(input=parent, child=node.name)
                key = (parent, node.name)
                assert parent in self._graph.keys()
                self._edges[key] = edge

    def add_node(self, node: Node):
        self._graph[node.name] = node
        self._total_nodes += 1

    def get_pa(self):
        pa = []
        for node in self._graph.values():
            if "IO" in node.op or len(node.input) == 0:
                continue
            pa.append(node.name)
        return pa

    def update(self, key: str):
        node = self._graph.pop(key)
        parents = []
        for edge in self._edges.keys():
            if node.name == edge[1]:
                parents.append(edge)
        children = []
        for edge in self._edges.keys():
            if node.name == edge[0]:
                children.append(edge)
        for parent in parents:
            for child in children:
                updated_jacobian = torch.sum(self._edges[parent].jacobian * self._edges[child].jacobian, dim=-1)
                try:
                    new_edge = self._edges[(parent[0], child[1])]
                    new_edge.jacobian += updated_jacobian
                    new_edge.count += 1
                except KeyError:
                    new_edge = Edge(input=parent[0], child=child[1], jacobian=updated_jacobian, count=1)
                    self._edges[(parent[0], child[1])] = new_edge
                self._graph[child[1]].input.add(parent[0])

        for edge in parents + children:
            self._edges.pop(edge)
        for child in children:
            self._graph[child[1]].input.remove(node.name)

        for k in self._edges.keys():
            assert node.name not in k
        return node

    def get_connectivity(self) -> np.ndarray:
        n = self._total_nodes
        # this is the extend jacobian
        out = np.zeros((n, n))
        for edge in self._edges.keys():
            r, c = edge
            r = self._graph[r].idx
            c = self._graph[c].idx
            out[r, c] += self._edges[edge].jacobian
        return out


def save_graph():
    nodes = []
    for idx, (node_name, edge) in enumerate(PATH.items()):
        op = "IO" if "y" in node_name else "no_op"
        nodes.append(Node(name=node_name, idx=idx, op=op, input=edge))
    with open("assets/basic_graph.pkl", "wb") as f:
        pickle.dump((nodes, {}), f)


def make_graph(fname) -> Graph:
    graph = Graph()
    with open(f"assets/{fname}.pkl", "rb") as f:
        nodes, _ = pickle.load(f)
    for idx, node in enumerate(nodes):
        graph.add_node(Node(name=node.name, idx=idx, input=node.input, op=node.op))
    graph.compute_edges()

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
        # TODO new_edges - old_edges now must become dependent on the degreee of sparisty
        # and the dimension of the muliplied jacobian
        # instead of a unitary "fill-in" we have a "fill-in" given by (n x k)  (k x m)
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
