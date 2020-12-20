from env import Env
from mcts import Evaluator, mcts
import numpy as np


class config:
    num_simulations = 10
    num_rollouts = 5
    uct_c = 1.25


def main():
    env = Env()
    simulator = Env()
    evaluator = Evaluator(game=simulator, n_rollouts=config.num_rollouts)
    state, _, done, info = env.reset()
    assert info["pa"]
    total_reward = 0
    t = 0
    actions = ["v2", "v1"]
    # env.render()
    while not done:
        #action = np.random.choice(info["pa"])
        action = mcts(env.get_state(), simulator, evaluator, config).best_child().action
        # action = actions[t]  # root.best_child().action
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(info)
        t += 1
        # actions.append(root.best_child().action)
    print(actions)
    print(total_reward)


if __name__ == '__main__':
    main()
