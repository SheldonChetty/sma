import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("Dataset/exp6.csv")

# Create Graph
G = nx.from_pandas_edgelist(df, 'source', 'target')

# ---------------- BASIC INFO ---------------- #
print("===== BASIC NETWORK INFO =====")
print("Number of Nodes:", G.number_of_nodes())
print("Number of Edges:", G.number_of_edges())
print("Density:", nx.density(G))

# ---------------- DEGREE ---------------- #
degree_dict = dict(G.degree())
print("\nDegree of Each Node:")
print(degree_dict)

# ---------------- CENTRALITY MEASURES ---------------- #
degree_centrality = nx.degree_centrality(G)
print("\nDegree Centrality:")
print(degree_centrality)

betweenness_centrality = nx.betweenness_centrality(G)
print("\nBetweenness Centrality:")
print(betweenness_centrality)

closeness_centrality = nx.closeness_centrality(G)
print("\nCloseness Centrality:")
print(closeness_centrality)

# Eigenvector (safe version)
try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    print("\nEigenvector Centrality:")
    print(eigenvector_centrality)
except:
    print("\nEigenvector Centrality did not converge")

# ---------------- BRIDGES ---------------- #
bridges = list(nx.bridges(G))
print("\nBridges in Network:")
print(bridges)

# ---------------- HUB NODE ---------------- #
hub_node = max(degree_dict, key=degree_dict.get)
print("\nHub Node:", hub_node)

# ---------------- VISUALIZATION ---------------- #
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.8, iterations=200, seed=42)

nx.draw_networkx_nodes(G, pos, node_size=500)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title("Game of Thrones Character Network")
plt.axis("off")
plt.show()
