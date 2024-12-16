#%%
""" get hcp results into a usable format"""
import pickle
terror_results = "./results/terror_configs.txt"
street_results = "./results/streets_configs.txt"
book_results = "./results/polbooks_configs.txt"
football_results = "./results/football_configs.txt"
airlines_results = "./results/airlines_configs.txt"

def read_data(path):
    with open(path,"rb") as f:
        data = pickle.load(f)
    return data
def save_data(data,path):
    with open(path,"wb") as f:
        pickle.dump(data,f)

def get_node_memberships(path):
    with open(path) as f:
        lines = f.readlines()[-1]
    final_config = list(map(int,lines.strip().split(' ')))
    memberships = list(map(decode_membership,final_config))
    return memberships
    # binary_nums = list(map(bin,final_config))

def decode_membership(num:int)->list:
    groups = []
    current_position=0
    while num>0:
        if num&1:
            groups.append(current_position)
        num>>=1
        current_position+=1
    return groups

def get_group_membersips(memberships):
    node_memberships_dict = {i:memberships[i] for i in range(len(memberships))}
    group_memberships = {}
    for node,groups in node_memberships_dict.items():
        for g in groups:
            group_memberships.setdefault(g,[]).append(node)
    return group_memberships

#%%
results = book_results
memberships_hcp = get_node_memberships(results)
group_memberships_hcp = get_group_membersips(memberships_hcp)
save_data(group_memberships_hcp,"data/hcp_books.pkl")

results = terror_results
memberships_hcp = get_node_memberships(results)
group_memberships_hcp = get_group_membersips(memberships_hcp)
save_data(group_memberships_hcp,"data/hcp_terror.pkl")


results = football_results
memberships_hcp = get_node_memberships(results)
group_memberships_hcp = get_group_membersips(memberships_hcp)
save_data(group_memberships_hcp,"data/hcp_football.pkl")

results = street_results
memberships_hcp = get_node_memberships(results)
group_memberships_hcp = get_group_membersips(memberships_hcp)
save_data(group_memberships_hcp,"data/hcp_street.pkl")


results = airlines_results
memberships_hcp = get_node_memberships(results)
group_memberships_hcp = get_group_membersips(memberships_hcp)
save_data(group_memberships_hcp,"data/hcp_airlines.pkl")
print("Done.")