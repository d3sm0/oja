from env import Env
from mcts import Evaluator, mcts


class config:
    num_simulations = 10
    num_rollouts = 5
    uct_c = 1.25


def main():
    env = Env(fname="graph")
    simulator = Env(fname="graph")
    evaluator = Evaluator(game=simulator, n_rollouts=config.num_rollouts)
    state, _, done, info = env.reset()
    actions = []
    assert info["pa"]
    total_reward = 0
    t = 0
    env.render()
    while not done:
        action = mcts(env.get_state(), simulator, evaluator, config).best_child().action
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(info)
        t += 1
        actions.append(action)
    print("Total number of operations required:", -total_reward)


if __name__ == '__main__':
    main()
