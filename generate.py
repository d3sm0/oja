import numpy as np

# generate random computational graph

np.random.seed(0)


def solve_dp(dptable, cost_fn, n_factors, n_features):
    for j in range(n_factors):
        for i in reversed(range(0, j + 1)):
            if i == j:
                dptable[j, i, 0] = n_features
                dptable[j, i, 1] = 0
            else:
                for k in range(i + 1, j + 1):
                    cost = dptable[j, k, 0] + dptable[k - 1, i, 0] + cost_fn[j, 0] * cost_fn[k, 1] * cost_fn[i, 1]
                    if k == i + 1 or cost < dptable[j, i, 0]:
                        dptable[j][i][0] = cost
                        dptable[j][i][1] = k

    pre_accumulation = 0
    for i in range(n_factors):
        pre_accumulation += cost_fn[i][2] * min(cost_fn[i])
    return pre_accumulation + dptable[n_factors - 1, 0, 0]


import enum


class Operation(enum.IntEnum):
    MM = 0
    GM = 1
    MG = 2


class Cost:
    output_dim = 0
    input_dim = 1
    n_edges = 2


def solve_gdp(dptable, cost_fn, n_factors):
    n_edges = 2
    n = 1
    m = 0
    MM = 0
    GM = 1
    MG = 2
    opt_cost = 0
    split_pos = 1
    opt_type = 2

    for j in range(n_factors):
        for i in reversed(range(0, j + 1)):
            if i == j:
                dptable[j, i, opt_cost] = cost_fn[j, n_edges] * min(cost_fn[j])
                dptable[j, i, split_pos] = 0
                if cost_fn[j, m] < cost_fn[j, n]:
                    dptable[j, i, opt_type] = MG
                else:
                    dptable[j, i, opt_type] = GM
            else:
                for k in range(i + 1, j + 1):
                    cost = dptable[j, k, opt_cost] + dptable[k - 1, i, opt_cost] + cost_fn[j, m] * cost_fn[k, n] * \
                           cost_fn[i, n]
                    if k == i + 1 or cost < dptable[j, i, opt_cost]:
                        # what does it mean split position
                        dptable[j, i] = (cost, k, MM) # here split changes

                    cost = dptable[k - 1, i, opt_cost]
                    depth = 0

                    for kk in range(k, j + 1):
                        depth += cost_fn[kk, n_edges]

                    cost += cost_fn[i, n] * depth

                    if cost < dptable[j, i, opt_cost]:
                        dptable[j, i, opt_cost] = cost
                        dptable[j, i, opt_type] = GM

                    cost = dptable[j, k, opt_cost]
                    depth = 0

                    for kk in range(i, k):
                        depth += cost_fn[kk, n_edges]

                    cost += cost_fn[j, m] * depth

                    if cost < dptable[j, i, opt_cost]:
                        dptable[j, i, opt_cost] = cost
                        dptable[j, i, opt_type] = MG
    # print(dptable)
    return dptable[n_factors - 1, 0, 0]


def test_graph():
    # output, input, number of edges
    graph = np.array([[3, 3, 29],
                      [1, 3, 14],
                      [2, 1, 7]])
    return graph


def print_table(table):
    n, m, _ = table.shape
    for j in range(0, n):
        for i in reversed(range(0, j + 1)):
            cost, split, operation = table[j, i]
            print(f"idx {j + 1, i + 1} \t cost: {cost} \t split: {split} \t operation:{operation}")


def main():
    depth = 3
    max_nm = 3
    n_features = 3
    n_actions = 3
    # graph = generate_graph(depth, max_nm)
    graph = test_graph()

    dptable = np.zeros(shape=(depth, depth, n_actions))
    dp_cost = solve_dp(dptable, graph, depth, n_features)

    dptable.fill(0)
    gdp_cost = solve_gdp(dptable, graph, depth)
    print_table(dptable)

    fwd_cost, reverse_cost = cost_fwd_reverse(graph, depth)

    print(f"dp_cost: {dp_cost} \t fwd_cost:{fwd_cost} \t reverse_cost:{reverse_cost}")


def cost_fwd_reverse(graph, n_factors):
    total = 0.
    for i in range(n_factors):
        total += graph[i, 2]
    m = 0
    n = 1
    fwd = graph[0, n] * total
    reverse = graph[n_factors - 1, m] * total
    return fwd, reverse


def generate_graph(depth, max_nm):
    graph = []
    print(depth)
    n = np.random.randint(1, max_nm)  # number of inputs
    m = np.random.randint(1, max_nm)  # number of ouptus
    n_e = np.random.randint(n + m, np.power(n + m, 2))  # n_edges
    print(m, n, n_e)
    graph.append((m, n, n_e))
    for i in range(1, depth):
        n = np.random.randint(1, max_nm)  # next input
        n_e = np.random.randint(n + m, np.power(n + m, 2))
        graph.append((m, n, n_e))
        # m is number of inputs at the current time fml !
        print((n, m, n_e))
        m = n

    graph = np.stack(graph)
    return graph


if __name__ == "__main__":
    main()
