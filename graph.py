import networkx as nx


class RandomGraph(object):
    def __init__(self, node_num, p, seed, k=4, m=5, graph_mode="ER"):
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.seed = seed
        self.graph_mode = graph_mode

        self.graph = self.make_graph()

    def make_graph(self):
        # reference
        # https://networkx.github.io/documentation/networkx-1.9/reference/generators.html

        if self.graph_mode is "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p, self.seed)
        elif self.graph_mode is "WS":
            graph = nx.random_graphs.watts_strogatz_graph(self.node_num, self.k, self.p, self.seed)
        elif self.graph_mode is "BA":
            graph = nx.random_graphs.barabasi_albert_graph(self.node_num, self.m, self.seed)

        return graph

    def get_graph_info(self):
        in_edges = {}
        in_edges[0] = []
        nodes = [0]
        end = []
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            # print(node, neighbors)

            edges = []
            check = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor + 1)
                    check.append(neighbor)
            if not edges:
                edges.append(0)
            in_edges[node + 1] = edges
            if check == neighbors:
                end.append(node + 1)
            nodes.append(node + 1)
        in_edges[self.node_num + 1] = end
        nodes.append(self.node_num + 1)

        # print(nodes, in_edges)
        return nodes, in_edges
