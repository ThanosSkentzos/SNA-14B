#%%
from data import get_terror,get_streets
import networkx as nx
G = get_terror()
nx.write_gml(G,"terror.gml")
nx.write_edgelist(G,"terror.txt")

S = get_streets()
nx.write_gml(S,"streets.gml")
nx.write_edgelist(S,"streets.txt")
# %%
print(G.number_of_nodes(),G.number_of_edges())
adj = G.adjacency()
# %%
