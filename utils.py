def plot_mcts(root, plot=True):
    """
    Plot the MCTS, pdf file is saved in the current directory.
    """
    try:
        from graphviz import Digraph
    except ModuleNotFoundError:
        print("Please install graphviz to get the MCTS plot.")
        return None

    graph = Digraph(comment="MCTS", engine="neato")
    graph.attr("graph", rankdir="LR", splines="true", overlap="false")
    id = 0

    def traverse(node, action, parent_id, best):
        nonlocal id
        node_id = id
        graph.node(
            str(node_id),
            label=node.__repr__(),
            color="orange" if best else "black",
        )
        id += 1
        if parent_id is not None:
            graph.edge(str(parent_id), str(node_id), constraint="false")

        if len(node.children) != 0:
            best_visit_count = max(
                [child.explore_count for child in node.children.values()]
            )
        else:
            best_visit_count = False
        for action, child in node.children.items():
            if child.explore_count != 0:
                traverse(
                    child,
                    action,
                    node_id,
                    True
                    if best_visit_count and child.explore_count == best_visit_count
                    else False,
                )

    traverse(root, None, None, True)
    graph.node(str(0), color="red")
    # print(graph.source)
    graph.render("mcts", view=False, cleanup=True, format="dot")
    return graph


def plot_computational_graph(graph):
    try:
        from graphviz import Digraph
    except ModuleNotFoundError:
        print("Please install graphviz to get the MCTS plot.")
        return None

    #from env import PATH
    digraph = Digraph(comment="graph", engine="dot")
    digraph.attr("graph", ranksep="equally", rank="same", rankdir="LR", ordering="out")
    for edges in graph:
        digraph.edge(*edges)
        #digraph.node(name=node)
        #child  = edges
        #if len(child):
        #    for c in child:
        #        digraph.edge(node, c)
        #if len(parent):
        #    for p in parent:
        #        digraph.edge(p, node)
    digraph.render("graph", view=True, format="pdf")
    print("")


    # order_graph.edge(style="invis")
    #for key in graph._vertexes.keys():
    #    # rank = get_rank(key)
    #    digraph.node(name=key, label=key, rank="same")
    #for (parent, child), value in graph._connectivity.items():
    #    if value == 1:
    #        digraph.edge(parent, child)

    #order_graph = Digraph(edge_attr={"style": "invis"}, node_attr={"style": "invis"})

    # order_graph.node("root", shape="record")
    # order_graph.node("", style="invis")
    #order_graph.edges((("x", "v"), ("v", "y")))

    #digraph.subgraph(order_graph)
    #order_graph.render(view=True)
    #return graphif

if __name__ == '__main__':
    plot_computational_graph()


