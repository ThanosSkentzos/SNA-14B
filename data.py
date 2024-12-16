# %%
import graph_tool.all as gt

import networkx as nx


def get_terror() -> nx.Graph:
    g = gt.collection.ns["terrorists_911"]
    edgelist = [(int(e.source()), int(e.target())) for e in list(g.edges())]
    G = nx.from_edgelist(edgelist)
    return G


def get_streets() -> nx.Graph:
    g = gt.collection.ns["urban_streets/ahmedabad"]
    edgelist = [(int(e.source()), int(e.target())) for e in list(g.edges())]
    G = nx.from_edgelist(edgelist)
    return G

def get_dataset(name)->nx.Graph:
    g = gt.collection.ns[name]
    edgelist = [(int(e.source()), int(e.target())) for e in list(g.edges())]
    G = nx.from_edgelist(edgelist)
    return G
#%%
import networkx as nx
G = get_terror()
nx.write_gml(G,"data/terror.gml")
nx.write_edgelist(G,"data/terror.txt")

S = get_streets()
nx.write_gml(S,"data/streets.gml")
nx.write_edgelist(S,"data/streets.txt")

name= "football"
G = get_dataset(name)
nx.write_gml(G,f"data/{name}.gml")
nx.write_edgelist(G,f"data/{name}.txt")

name= "polbooks"
G = get_dataset(name)
nx.write_gml(G,f"data/{name}.gml")
nx.write_edgelist(G,f"data/{name}.txt")

name= "eu_airlines"
G = get_dataset(name)
nx.write_gml(G,f"data/{name}.gml")
nx.write_edgelist(G,f"data/{name}.txt")
# %%
