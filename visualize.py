# %%
import numpy as np
def get_node_from_group(group_memberships):
    node_memberships={}
    groups = list(group_memberships.keys())
    for group, nodes in group_memberships.items():
        for node in nodes:
            node_memberships.setdefault(node,[]).append(groups.index(group))
    return node_memberships.values()

def plot_results(node_memberships,graph,title,pos):
    top_membership = [max(memb_list) for memb_list in node_memberships]
    num_groups = max(top_membership) + 1 
    colors = plt.cm.tab10(range(num_groups))  # Generate a set of colors
    if num_groups>12:
        colors = plt.cm.tab20(range(num_groups))  # Generate a set of colors
    node_colors = [colors[t] for t in top_membership]
    patches = [mpatches.Patch(color=colors[g],label=f"Group {g}") for g in range(num_groups)]
    nx.draw(graph,node_size=42,node_color=node_colors,pos=pos)
    plt.legend(handles=patches,title=title)
#%%

from data import get_terror,get_streets,get_dataset
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from extract_hcp import read_data
#%%
# political books
np.random.seed(42)
G = get_dataset("polbooks")

group_memberships_hcp = read_data("data/hcp_books.pkl")
memberships_hcp = get_node_from_group(group_memberships_hcp)

group_memberships_sbm = read_data("data/Gallagher_Layered_nodes.pkl")
memberships_sbm = get_node_from_group(group_memberships_sbm)

group_memberships_kcore = read_data("data/KCore_nodes.pkl")
memberships_kcore = get_node_from_group(group_memberships_kcore)

group_memberships_cope = read_data("data/BnE_nodes.pkl")
memberships_cope = get_node_from_group(group_memberships_cope)

pos = nx.spring_layout(G)
plot_results(memberships_hcp,G,"Book groups hcp",pos)
plt.figure()
plot_results(memberships_sbm,G,"Book groups sbm",pos)
plt.figure()
plot_results(memberships_kcore,G,"Book groups kcore",pos)
plt.figure()
plot_results(memberships_cope,G,"Book groups core-per",pos)
#%%
# terror
G = get_terror()
np.random.seed(42)

group_memberships_hcp = read_data("data/hcp_terror.pkl")
memberships_hcp = get_node_from_group(group_memberships_hcp)

group_memberships_sbm = read_data("data/Gallagher_Layered_terro.pkl")
memberships_sbm = get_node_from_group(group_memberships_sbm)

group_memberships_kcore = read_data("data/KCore_terro.pkl")
memberships_kcore = get_node_from_group(group_memberships_kcore)

group_memberships_cope = read_data("data/BnE_terro.pkl")
memberships_cope = get_node_from_group(group_memberships_cope)

pos = nx.spring_layout(G)
plot_results(memberships_hcp,G,"Terror hcp",pos)
plt.figure()
plot_results(memberships_sbm,G,"Terror sbm",pos)
plt.figure()
plot_results(memberships_kcore,G,"Terror kcore",pos)
plt.figure()
plot_results(memberships_cope,G,"Terror core-per",pos)
# %%
# football
G = get_dataset("football")
np.random.seed(42)

group_memberships_hcp = read_data("data/hcp_football.pkl")
memberships_hcp = get_node_from_group(group_memberships_hcp)

group_memberships_sbm = read_data("data/Gallagher_Layered_football.pkl")
memberships_sbm = get_node_from_group(group_memberships_sbm)

group_memberships_kcore = read_data("data/KCore_football.pkl")
memberships_kcore = get_node_from_group(group_memberships_kcore)

group_memberships_cope = read_data("data/BnE_football.pkl")
memberships_cope = get_node_from_group(group_memberships_cope)

pos = nx.spring_layout(G)
plot_results(memberships_hcp,G,"Football hcp",pos)
plt.figure()
plot_results(memberships_sbm,G,"Football sbm",pos)
plt.figure()
plot_results(memberships_kcore,G,"Football kcore",pos)
plt.figure()
plot_results(memberships_cope,G,"Football core-per",pos)
#%%
G = get_streets()
np.random.seed(42)

group_memberships_hcp = read_data("data/hcp_street.pkl")
memberships_hcp = get_node_from_group(group_memberships_hcp)

# # group_memberships_sbm = read_data("data/Gallagher_Layered_football.pkl")
# # memberships_sbm = get_node_from_group(group_memberships_sbm)

# # group_memberships_kcore = read_data("data/KCore_football.pkl")
# # memberships_kcore = get_node_from_group(group_memberships_kcore)

# # group_memberships_cope = read_data("data/BnE_football.pkl")
# # memberships_cope = get_node_from_group(group_memberships_cope)

pos = nx.spring_layout(G)
plot_results(memberships_hcp,G,"Streets hcp",pos)
# plt.figure()
# plot_results(memberships_sbm,G,"Football sbm",pos)
# plt.figure()
# plot_results(memberships_kcore,G,"Football kcore",pos)
# plt.figure()
# plot_results(memberships_cope,G,"Football core-per",pos)
#%%
# airlines
G = get_dataset("eu_airlines")
np.random.seed(42)

group_memberships_hcp = read_data("data/hcp_airlines.pkl")
memberships_hcp = get_node_from_group(group_memberships_hcp)

group_memberships_sbm = read_data("data/Gallagher_Layered_eu_airlines.pkl")
memberships_sbm = get_node_from_group(group_memberships_sbm)

group_memberships_kcore = read_data("data/KCore_eu_airlines.pkl")
memberships_kcore = get_node_from_group(group_memberships_kcore)

group_memberships_cope = read_data("data/BnE_eu_airlines.pkl")
memberships_cope = get_node_from_group(group_memberships_cope)

pos = nx.spring_layout(G)
plot_results(memberships_hcp,G,"Airlines hcp",pos)
plt.figure()
plot_results(memberships_sbm,G,"Airlines sbm",pos)
plt.figure()
plot_results(memberships_kcore,G,"Airlines kcore",pos)
plt.figure()
plot_results(memberships_cope,G,"Airlines core-per",pos)
# %%
