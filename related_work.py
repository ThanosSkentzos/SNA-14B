import graph_tool.all as gt
import networkx as nx
import numpy as np
import math
from itertools import combinations
from scipy.optimize import minimize
from core_periphery_sbm import core_periphery as cp

# Function to convert graph-tool graph to NetworkX graph
def graph_tool_to_networkx(gt_graph):
    nx_graph = nx.Graph() if not gt_graph.is_directed() else nx.DiGraph()
    for v in gt_graph.vertices():
        nx_graph.add_node(int(v))
    for e in gt_graph.edges():
        nx_graph.add_edge(int(e.source()), int(e.target()))
    return nx_graph

# Generalized function to calculate P(A | k, g)
def calculate_p_a_k_g(graph, labels, mode="coreness", return_log=False):
    """
    Generalized function to calculate P(A | k, g) for various core-periphery methods.
    Includes handling for large m and t using logarithmic factorials.
    """
    from math import log, exp
    def log_factorial(n):
        return sum(log(i) for i in range(1, n + 1))

    adjacency_matrix = nx.to_numpy_array(graph)
    nodes = list(graph.nodes())

    if mode == "coreness":
        # Coreness-based (Borgatti-Everett or k-Core)
        coreness = list(labels.values())
        pattern_matrix = np.outer(coreness, coreness)
        m = np.sum(adjacency_matrix * pattern_matrix)
        t = np.sum(pattern_matrix)
    elif mode in ["group", "monte_carlo"]:
        # Group-based (Gallagher or Monte Carlo)
        k = len(set(labels.values()))
        m, t = [0] * k, [0] * k
        layer_to_nodes = {r: [] for r in range(k)}
        for node, group in labels.items():
            layer_to_nodes[group].append(node)
        for r in range(k):
            nodes_in_layer = layer_to_nodes[r]
            if len(nodes_in_layer) < 2:
                continue
            m[r] = sum(1 for u, v in combinations(nodes_in_layer, 2) if graph.has_edge(u, v))
            t[r] = len(nodes_in_layer) * (len(nodes_in_layer) - 1) // 2

        # For group/Monte Carlo, use factorial logic over layers
        p_a_k_g = 1.0
        for r in range(k):
            if t[r] > 0:
                if t[r] > 1000 or m[r] > 1000:  # Logarithmic factorial for large values
                    if m[r] > 0 and t[r] > m[r]:
                        P_log = log_factorial(int(m[r])) + log_factorial(int(t[r] - m[r])) - log_factorial(int(t[r] + 1))
                        p_a_k_g *= exp(P_log)
                        return p_a_k_g, "log P(A | k, g)"
                else:
                    p_a_k_g *= (math.factorial(m[r]) * math.factorial(t[r] - m[r])) / math.factorial(t[r] + 1)
                    return p_a_k_g, "P(A | k, g)"
    else:
        raise ValueError("Unsupported mode. Choose 'coreness', 'group', or 'monte_carlo'.")

    # For coreness mode, use general factorial logic
    if t > 0:
        if t > 1000 or m > 1000:  # Logarithmic factorial for large values
            if m > 0 and t > m:
                P_log = log_factorial(int(m)) + log_factorial(int(t - m)) - log_factorial(int(t + 1))
                return (P_log if return_log else exp(P_log), "log P(A | k, g)" if return_log else "P(A | k, g)")
        else:
            return ((math.factorial(int(m)) * math.factorial(int(t - m))) / math.factorial(int(t + 1)), "P(A | k, g)")
    return 0


# Borgatti-Everett Method
def borgatti_everett_core_periphery(graph):
    nodes = list(graph.nodes())
    adjacency_matrix = nx.to_numpy_array(graph)
    coreness = np.random.rand(len(nodes))

    def pattern_matrix(c):
        return np.outer(c, c)

    def objective_function(c):
        p_matrix = pattern_matrix(c)
        correlation = np.corrcoef(adjacency_matrix[np.triu_indices(len(nodes), k=1)],
                                  p_matrix[np.triu_indices(len(nodes), k=1)])[0, 1]
        return -correlation

    result = minimize(objective_function, coreness, bounds=[(0, 1)] * len(nodes), method="L-BFGS-B")
    return dict(zip(nodes, result.x))

# k-Core Decomposition
def k_core_decomposition(graph):
    coreness = nx.core_number(graph)
    max_coreness = max(coreness.values())
    core_nodes = [node for node, core in coreness.items() if core == max_coreness]
    periphery_nodes = [node for node, core in coreness.items() if core < max_coreness]
    return coreness, core_nodes, periphery_nodes


# Function to process dataset
def process_dataset(dataset_name):
    G_gt = gt.collection.ns[dataset_name]
    G = graph_tool_to_networkx(G_gt)

    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")

    # Apply Borgatti-Everett Method
    node2coreness_bne = borgatti_everett_core_periphery(G)
    threshold_bne = np.median(list(node2coreness_bne.values()))
    core_nodes_bne = [node for node, coreness in node2coreness_bne.items() if coreness > threshold_bne]
    periphery_nodes_bne = [node for node, coreness in node2coreness_bne.items() if coreness <= threshold_bne]
    p_a_k_g_bne, result_type_bne = calculate_p_a_k_g(G, node2coreness_bne, mode="coreness", return_log=True)
    print("----------------BORGATTI AND EVERETT----------------")
    # Threshold coreness to identify core and periphery nodes
    threshold = np.median(list(node2coreness_bne.values()))
    print(f"Median Coreness Threshold: {threshold}")
#   print("Node Coreness Values (BNE):", node2coreness_bne)
    print("Core Nodes (BNE):", core_nodes_bne)
    print("Periphery Nodes (BNE):", periphery_nodes_bne)

    # Apply Gallagher's Method
    # Initialize and infer Hub-and-Spoke model
    hubspoke = cp.HubSpokeCorePeriphery(n_gibbs=100, n_mcmc=100)
    hubspoke.infer(G)

    # Initialize and infer Layered model
    if dataset_name == "football":
        n_layers=14
    elif dataset_name == "eu_airlines":
        n_layers=24
    else:
        n_layers=3

    # Initialize and infer Layered model
    layered = cp.LayeredCorePeriphery(n_layers, n_gibbs=100, n_mcmc=100)
    layered.infer(G)

    # Get node-to-group assignments for both models
    node2label_hs = hubspoke.get_labels(prob=False, return_dict=True)
    node2label_l = layered.get_labels(prob=False, return_dict=True)
    node2probs_hs = hubspoke.get_labels(prob=True, return_dict=True)
    node2probs_l = layered.get_labels(prob=True, return_dict=True)

    # Compute layer connection probabilities manually
    k = len(set(node2label_l.values()))  # Number of layers
    adjacency_matrix = nx.to_numpy_array(G)
    layer_probs = {}

    # Calculate P(A | k, g) for both models
    p_a_k_g_hs, result_type_hs = calculate_p_a_k_g(G, node2label_hs, mode="group", return_log=True)
    p_a_k_g_l, result_type_l = calculate_p_a_k_g(G, node2label_l, mode="group", return_log=True)

    # Determine best Gallagher structure
    best_gallagher = node2label_hs if p_a_k_g_hs > p_a_k_g_l else node2label_l
    if p_a_k_g_hs > p_a_k_g_l:
        p_a_k_g_gallagher = p_a_k_g_hs
        result_type_gallagher = result_type_hs
        best_gallagher_type = "Hub-Spoke"
    else:
        p_a_k_g_gallagher = p_a_k_g_l
        result_type_gallagher = result_type_l
        best_gallagher_type = "Layered"
    print("---------------------------------------------------------------------")
    print("--------------------GALLAGHER--------------------")
    print("The suitable structure is ", best_gallagher_type)
    if best_gallagher_type == "Hub-Spoke":
        # Print probabilities for Hub-and-Spoke model
        print("Node to probabilities (HubSpoke):", node2probs_hs)
        # Print core and periphery nodes for Hub-and-Spoke model
        core_nodes_hs = [node for node, group in node2label_hs.items() if group == 0]  # Assuming core is group 0
        periphery_nodes_hs = [node for node, group in node2label_hs.items() if group != 0]
        print("Core Nodes (HubSpoke):", core_nodes_hs)
        print("Periphery Nodes (HubSpoke):", periphery_nodes_hs)
        core_set_gallagher = set(core_nodes_hs)
    else:
        # Print probabilities for Layered model
#       print("Node to probabilities (Layered):", node2probs_l)
        # Count and print number of layers detected in the Layered model
        core_nodes_l = [node for node, group in node2label_l.items() if group == 0]  # Assuming core is group 0
        periphery_nodes_l = [node for node, group in node2label_l.items() if group != 0]
        core_set_gallagher = set(core_nodes_l)
        num_layers = len(set(node2label_l.values()))
        print(f"Number of layers detected in the Layered structure: {num_layers}")
        # Group nodes by layer
        layer_to_nodes = {layer: [] for layer in range(k)}
        for node, layer in node2label_l.items():
            layer_to_nodes[layer].append(node)

        # Calculate probabilities for each layer
        for layer, nodes in layer_to_nodes.items():
            if len(nodes) > 1:  # Avoid empty or single-node layers
                subgraph_matrix = adjacency_matrix[np.ix_(nodes, nodes)]
                edge_count = np.sum(subgraph_matrix) / 2  # Undirected graph, so divide by 2
                total_possible_edges = len(nodes) * (len(nodes) - 1) / 2
                layer_probs[layer] = edge_count / total_possible_edges if total_possible_edges > 0 else 0
            else:
                layer_probs[layer] = 0  # No edges possible in single-node layers

        # Sort layers based on connection probabilities
        sorted_layers = sorted(layer_probs.items(), key=lambda x: x[1], reverse=True)
        core_layer = sorted_layers[0][0]  # Layer with the highest connection probability
        print(f"Core layer identified based on probabilities: Layer {core_layer}")

        # Classify nodes into core and periphery based on identified core layer
        core_nodes_l = [node for node, group in node2label_l.items() if group == core_layer]
        periphery_nodes_l = [node for node, group in node2label_l.items() if group != core_layer]

        # Display all layers with their probabilities and node assignments
        print("Layer connection probabilities and node assignments:")
        for layer, prob in sorted_layers:
            layer_nodes = [node for node, group in node2label_l.items() if group == layer]
            layer_type = "Core" if layer == core_layer else "Periphery"
            print(f"Layer {layer} ({layer_type}): Probability = {prob:.4f}, Nodes = {layer_nodes}")


    # Apply k-Core Decomposition
    coreness_kcore, core_nodes_kcore, periphery_nodes_kcore = k_core_decomposition(G)
    p_a_k_g_kcore, result_type_kcore = calculate_p_a_k_g(G, coreness_kcore, mode="coreness", return_log=True)
    print("--------------------------------------------------------------------")
    print("---------------K-CORE DECOMPOSITION--------------")
#   print("Node Coreness Values (k-Core):", coreness_kcore)
    print("Core Nodes (k-Core):", core_nodes_kcore)
    print("Periphery Nodes (k-Core):", periphery_nodes_kcore)

    # Overlapping Calculations
    jaccard_bne_kcore = len(set(core_nodes_bne) & set(core_nodes_kcore)) / len(set(core_nodes_bne) | set(core_nodes_kcore))

    # Jaccard Index for BNE vs Gallagher
    jaccard_bne_gallagher = len(set(core_nodes_bne) & core_set_gallagher) / len(set(core_nodes_bne) | core_set_gallagher)

    # Jaccard Index for k-Core vs Gallagher
    jaccard_kcore_gallagher = len(set(core_nodes_kcore) & core_set_gallagher) / len(set(core_nodes_kcore) | core_set_gallagher)



    # Print Final Results
    print("---------------------------------------------------------------------")
    print("--------------------- RESULT --------------------")
    print(f"{result_type_bne} for Bogatti: {p_a_k_g_bne}")
    print(f"{result_type_gallagher} for Gallagher ({best_gallagher_type}): {p_a_k_g_gallagher}")
    print(f"{result_type_kcore} for k-Core: {p_a_k_g_kcore}")
    print(f"Jaccard Index (Bogatti vs k-Core): {jaccard_bne_kcore}")
    print(f"Jaccard Index (Bogatti vs Gallagher): {jaccard_bne_gallagher}")
    print(f"Jaccard Index (k-Core vs Gallagher): {jaccard_kcore_gallagher}")


    import pickle

    # Function to save nodes into pickle files
    def save_to_pickle(data, filename):
        if 'street' in filename or 'street' in data:
            print(80*'-')
            print('Skipping streets')
            return
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print("--------------------------------------------------------------------")
        print(f"Data saved to {filename}")

    # Convert and save BNE core and periphery nodes
    core_bne_list = list(core_nodes_bne)
    periphery_bne_list = list(periphery_nodes_bne)
    save_to_pickle({"core": core_bne_list, "periphery": periphery_bne_list}, f"data/BnE_{dataset_name}.pkl")

    if best_gallagher_type == "Hub-Spoke":
        # Convert and save Gallagher's Hub-and-Spoke core and periphery nodes
        core_hs_list = list(core_nodes_hs)
        periphery_hs_list = list(periphery_nodes_hs)
        save_to_pickle({"core": core_hs_list, "periphery": periphery_hs_list}, f"data/Gallagher_HubSpoke_{dataset_name}.pkl")
    else:
        # Convert and save Gallagher's Layered structure nodes (each layer)
        layered_nodes = {}
        for layer, nodes in layer_to_nodes.items():
            layered_nodes[f"layer_{layer}"] = nodes
        save_to_pickle(layered_nodes, f"data/Gallagher_Layered_{dataset_name}.pkl")

    # Convert and save K-Core core and periphery nodes
    core_kcore_list = list(core_nodes_kcore)
    periphery_kcore_list = list(periphery_nodes_kcore)
    save_to_pickle({"core": core_kcore_list, "periphery": periphery_kcore_list}, f"data/KCore_{dataset_name}.pkl")


    # print("All data has been converted and saved to pickle files.")


    # Function to load pickle file
    def load_pickle_file(filepath):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        print("--------------------------------------------------------------------")
        print(f"Pickle file loaded successfully: {filepath}")
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Data keys: {list(data.keys())[:10]} (showing up to 10 keys)")
            print(f"Total keys: {len(data)}")
        return data

    # Function to calculate Jaccard Index and NMI
    def calculate_jaccard(set1, set2, total_nodes):
        set1, set2 = set(set1), set(set2)
        # Jaccard Index
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0
        return jaccard

    # Function to compare Monte Carlo with methods
    def compare_mc_with_methods(mc_layers, method_core_nodes, method_name, total_nodes):
        jaccard_scores = []
        for mc_layer, mc_nodes in mc_layers.items():
            jaccard = calculate_jaccard(mc_nodes, method_core_nodes, total_nodes)
            jaccard_scores.append(jaccard)
        # Average Jaccard across layers
        avg_jaccard = np.mean(jaccard_scores)
        print(f"Jaccard Index (MC vs {method_name}): {avg_jaccard}")
        return avg_jaccard

    # Function to compare Monte Carlo with Gallagher Layered structure
    def compare_mc_with_gallagher(mc_layers, gallagher_layers, total_nodes):
        jaccard_scores = []

        for g_layer, g_nodes in gallagher_layers.items():
            for mc_layer, mc_nodes in mc_layers.items():
                jaccard = calculate_jaccard(mc_nodes, g_nodes, total_nodes)
                jaccard_scores.append(jaccard)
        # Average Jaccard across all layer comparisons
        avg_jaccard = np.mean(jaccard_scores)
        print(f"Jaccard Index (MC vs Gallagher): {avg_jaccard}")
        return avg_jaccard

    # Step 1: Load Monte Carlo pickle file
    if dataset_name == "terrorists_911":
        mc_file_path = "data/hcp_terror.pkl" 
    elif dataset_name == "polbooks":
        mc_file_path = "data/hcp_books.pkl" 
    elif dataset_name == "football":
        mc_file_path = "data/hcp_football.pkl" 
    elif dataset_name == "urban_streets/ahmedabad":
        mc_file_path = "data/hcp_streets.pkl"
    else:
        mc_file_path = "data/hcp_airlines.pkl"
    monte_carlo_data = load_pickle_file(mc_file_path)


    # Monte Carlo vs BnE
    print("--------------------------------------------------------------------")
    jaccard_bne_mc = calculate_jaccard(set(core_nodes_bne), monte_carlo_data[0], len(G.nodes()))
    print(f"Jaccard Index (Monte Carlo vs Bogatti): {jaccard_bne_mc}")

    # Monte Carlo vs k-Core
    jaccard_kcore_mc = calculate_jaccard(set(core_nodes_kcore), monte_carlo_data[0], len(G.nodes()))
    print(f"Jaccard Index (Monte Carlo vs k-Core): {jaccard_kcore_mc}")

    # Monte Carlo vs Gallagher Layered
    print("--------------------------------------------------------------------")
    print("Monte Carlo vs Gallagher Layered:")
    for layer in monte_carlo_data.keys():
        if layer in layer_to_nodes:  # Ensure the layer exists in Gallagher
            jaccard_gallagher_mc = calculate_jaccard(monte_carlo_data[layer], layer_to_nodes[layer], len(G.nodes()))
            print(f"Layer {layer} vs Layer {layer}: Jaccard Index = {jaccard_gallagher_mc}")
            

process_dataset("terrorists_911")
process_dataset("polbooks")
process_dataset("football")
process_dataset("urban_streets/ahmedabad")
process_dataset("eu_airlines")