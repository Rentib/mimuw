import networkx as nx

# Create a new graph
G = nx.Graph()

# Add isolated node on the left
isolated_node = "0"
G.add_node(isolated_node)

# Add Clique-6 on the right
clique_nodes = [i for i in range(1, 8)]
G.add_nodes_from(clique_nodes)
G.add_edges_from((n1, n2) for n1 in clique_nodes for n2 in clique_nodes if n1 != n2)

# Positioning: isolated node on the left, Clique-6 on the right
pos = {isolated_node: (-5, 0)}
pos.update(nx.circular_layout(clique_nodes, scale=1.5, center=(1, 0)))

# Draw the graph
# nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", edge_color="gray", font_size=10, font_weight="bold")

G_complement = nx.complement(G)

print(
    nx.to_latex(
        G_complement,
        pos,
        as_document=False,
        default_node_options="every node/.style={circle, draw}",
    )
)

print(
    nx.to_latex(
        G,
        pos,
        as_document=False,
        default_node_options="every node/.style={circle, draw}",
    )
)
