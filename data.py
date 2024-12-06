# %%
import graph_tool.all as gt

import networkx as nx


def get_terror():
    g = gt.collection.ns["terrorists_911"]
    edgelist = [(int(e.source()), int(e.target())) for e in list(g.edges())]
    G = nx.from_edgelist(edgelist)
    return G


# %%
