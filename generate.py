# generate random computational graph
import enum
# each vertex has input size output size and n_edges
import typing

import numpy as np


class Operation(enum.IntEnum):
    accumulate = 0  # accumulate
    tangent = 1  # forward
    adjoint = 2  # backward


class Cost:
    output_dim = 0
    input_dim = 1
    n_edges = 2


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


# aux = 0 for cost, 1 for split, 2 action
# state = (j, i, aux)


def test_graph():
    # Key assumption: within a layer or module the dimension do not change
    # elemental operation are "element wise" so they are applied at every element of the jacobian
    # output_size, input_size, number of edges
    # number of edges is the number of elemental operations inside the ith layer
    # for example linear layer n_edges = 3: dot,sum, relu, and a total cost of n * m
    graph = np.array([[3, 3, 29],
                      [1, 3, 14],
                      [2, 1, 7]])
    return graph


class Vertex(typing.NamedTuple):
    n = 0
    m = 0
    n_edges = 0


def solve_gdp(dptable, cost_fn, depth):
    n_edges = 2
    input_slice = 1
    output_slice = 0
    opt_cost = 0
    split_pos = 1
    action_slice = 2

    for parent in range(depth):
        for child in reversed(range(0, parent + 1)):
            if child == parent:
                # here using min is scale indpendent, so we can pull it out, but in reality min is a "decision" not a feature of the cost function
                dptable[parent, child, opt_cost] = cost_fn[parent, n_edges] * min(cost_fn[parent, output_slice],
                                                                                  cost_fn[child, input_slice])
                dptable[parent, child, split_pos] = 0
                if cost_fn[parent, output_slice] < cost_fn[parent, input_slice]:
                    dptable[parent, child, action_slice] = Operation.adjoint
                else:
                    dptable[parent, child, action_slice] = Operation.tangent
            else:
                for successor in range(child + 1, parent + 1):
                    # evaluate terminal
                    # set number of edges to 1
                    cost = dptable[parent, successor, opt_cost] + dptable[successor - 1, child, opt_cost]
                    cost += cost_fn[parent, output_slice] * cost_fn[successor, input_slice] * cost_fn[
                        child, input_slice]
                    if successor == child + 1 or cost < dptable[parent, child, opt_cost]:
                        # what does it mean split position
                        dptable[parent, child] = (cost, successor, Operation.accumulate)

                        # evaluate for forward
                    # current reward
                    cost = dptable[successor - 1, child, opt_cost]
                    depth = 0
                    # future values
                    for kk in range(successor, parent + 1):
                        depth += cost_fn[kk, n_edges]

                    cost += cost_fn[child, input_slice] * depth
                    # acceptance condition
                    if cost < dptable[parent, child, opt_cost]:
                        dptable[parent, child, opt_cost] = cost
                        dptable[parent, child, action_slice] = Operation.tangent

                    # evaluate for backward
                    cost = dptable[parent, successor, opt_cost]
                    depth = 0

                    for kk in range(child, successor):
                        depth += cost_fn[kk, n_edges]

                    cost += cost_fn[parent, output_slice] * depth

                    if cost < dptable[parent, child, opt_cost]:
                        dptable[parent, child, opt_cost] = cost
                        dptable[parent, child, action_slice] = Operation.adjoint
    # print(dptable)
    return dptable[depth - 1, 0, 0]


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
