# from __future__ import annotations

import os
from typing import Tuple, List, Union, Text

import gym
import numpy as np

from env import GameState


class Evaluator:
    def __init__(self, game: gym.Env, n_rollouts: int = 10, seed: int = 0):
        self.game = game
        self.n_rollouts = n_rollouts
        self._prior = np.ones((self.game.action_space.n,)) / self.game.action_space.n
        self.rs = np.random.default_rng(seed)

    def evaluate(self, state: GameState) -> float:
        value = 0
        if not state.done:
            for n in range(self.n_rollouts):
                game_state = self.game.set_state(state)
                info = game_state.info
                done = game_state.done
                while not done:
                    action = self.rs.choice(info["pa"])
                    _, reward, done, info = self.game.step(action)
                    value += reward

        return value / self.n_rollouts

    @property
    def prior(self) -> np.ndarray:
        return self._prior


class Node:
    def __init__(self, action: Union[int, None], prior: float = 1.0):
        self.total_reward = 0.0
        self.reward = 0.0
        self.explore_count = 0
        self.children = {}
        self.prior = prior
        self.action = action

    def uct(self, parent_count: int, uct_c: float = 1.25) -> float:
        if self.explore_count == 0:
            return np.inf
        return self.value() + uct_c * np.sqrt(np.log(parent_count) / self.explore_count)

    def value(self) -> float:
        if self.explore_count == 0:
            return 0
        return self.reward + self.total_reward / self.explore_count

    def sort_key(self) -> Tuple[float, int, float]:
        return self.value(), self.explore_count, self.total_reward

    def best_child(self) -> Node:
        return max(self.children.values(), key=Node.sort_key)

    def __repr__(self) -> Text:
        return "A:{}, V:{}, #:{}".format(self.action, self.value(), self.explore_count)


from utils import plot_mcts


def mcts(state, simulator, evaluator, config):
    root = Node(action=None)
    # epxand root + dirichile noise
    for n in range(config.num_simulations):
        simulator.set_state(state)
        path = select(root, simulator)
        value = evaluator.evaluate(simulator.get_state())
        backpropagate(path, value)
    plot_mcts(root, fname=os.path.join("assets", f"mcts_{len(root.children)}"))
    return root


def expand(node, state):
    for action in state.info["pa"]:
        node.children[action] = Node(action=action, prior=1)


def backpropagate(path: List[Node], value: float):
    for node in reversed(path):
        node.total_reward = value
        node.explore_count += 1
        value = node.reward + value


def select(root: Node, game: gym.Env, uct_c: float = 1.25) -> List[Node]:
    path = [root]
    node = root
    while node.explore_count > 0:
        if not node.children:
            expand(node, game.get_state())
        node = max(
            node.children.values(),
            key=lambda c: Node.uct(c, node.explore_count, uct_c)
        )
        _, reward, done, info = game.step(node.action)
        node.reward = reward
        path.append(node)
        if done:
            break

    return path
