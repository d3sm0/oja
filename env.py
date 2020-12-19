from __future__ import annotations

import collections
import copy
from typing import NamedTuple, Dict, Any, Tuple, Union, List

import gym
import numpy as np

from utils import plot_computational_graph

PATH = {"x1": ("v1", "y3"),
        "x2": ("v1",),
        "x3": ("v1",),
        "v1": ("v2", "y3"),
        "v2": ("y1", "y2", "y3"),
        "y1": (),
        "y2": (),
        "y3": (),
        }


class GameState(NamedTuple):
    state: np.ndarray
    # reward: float
    done: bool
    info: Dict[str, Union[List, Graph]]


class Node(NamedTuple):
    key: str
    idx: int
    parent: set = set()
    children: set = set()


class Graph:
    def __init__(self):
        self._graph = {}
        self._edges = []
        self._total_nodes = 0
        self._cost = collections.defaultdict(lambda: 0)

    def add_node(self, node: Node):
        self._graph[node.key] = node
        self._total_nodes += 1
        for s in node.children:
            self._edges.append((node.key, s))

    def get_pa(self):
        pa = []
        for node in self._graph.values():
            if "v" in node.key:
                pa.append(node.key)
        return pa

    def update(self, key: str):
        node = self._graph.pop(key)
        in_edge = set()
        out_edge = set()
        for edge in self._edges:
            if node.key == edge[0]:
                out_edge.add(edge[1])
            elif node.key == edge[1]:
                in_edge.add(edge[0])
            else:
                continue
        for parent in in_edge:
            self._edges.remove((parent, key))
            for child in out_edge:
                try:
                    self._edges.remove((key, child))
                except ValueError:
                    pass
                self._edges.append((parent, child))
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


def make_graph() -> Graph:
    graph = Graph()
    for idx, k in enumerate(PATH.keys()):
        node = Node(k, idx=idx, children=set(PATH[k]))
        graph.add_node(node)
    return graph


class Env(gym.Env):
    def __init__(self):
        self.graph = make_graph()
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
        self.graph = make_graph()

        m0 = self.graph.get_connectivity()
        info = {
            "pa": self.graph.get_pa()
        }

        return m0, 0, False, info

    def render(self, mode="human"):
        plot_computational_graph(self.graph._edges)

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
