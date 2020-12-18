from env import Env
from mcts import Evaluator


class config:
    num_simulations = 50
    num_rollouts = 100
    uct_c = 1.25


def main():
    env = Env()
    simulator = Env()
    evaluator = Evaluator(game=simulator, n_rollouts=config.num_rollouts)
    state, _, done, info = env.reset()
    assert info["pa"]
    total_reward = 0
    actions = []
    t = 0
    actions = ["v2", "v3", "v1"]
    while not done:
        # action = np.random.choice(info["pa"])
        # root = mcts(env.get_state(), simulator, evaluator, config)
        action = actions[t]  # root.best_child().action
        state, reward, done, info = env.step(action)
        total_reward += reward
        print(info)
        t += 1
        # actions.append(root.best_child().action)
    print(actions)
    print(total_reward)


if __name__ == '__main__':
    main()
