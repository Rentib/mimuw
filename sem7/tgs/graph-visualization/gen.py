import matplotlib.pyplot as plt
import networkx as nx
import os

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # Use pdflatex for processing
    "font.family": "serif",       # Use serif font (matches LaTeX default)
    "text.usetex": True,          # Use LaTeX for text rendering
})

def draw_graph(G, pos, filename):
    nx.draw(G,
            pos,
            with_labels=True,
            node_color='white',
            edgecolors='black',
            node_size=1000,
            edge_color='black')
    plt.savefig(filename)
    plt.close()

def gen_fig1():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.bfs_layout(G, 1)
    draw_graph(G, pos, "./figures/fig1.pgf")

def gen_bipartite():
    G = nx.bipartite.gnmk_random_graph(4, 5, 11, seed=69)
    top = nx.bipartite.sets(G)[0]
    pos = nx.bipartite_layout(G, top)
    draw_graph(G, pos, "./figures/bipartite.pgf")

def gen_bfs():
    G = nx.gnm_random_graph(10, 15, seed=420)
    pos = nx.bfs_layout(G, 0)
    draw_graph(G, pos, "./figures/bfs.pgf")

def gen_circular():
    G = nx.cycle_graph(10)
    G.add_edges_from([(0, 5), (1, 6), (4, 7)])
    pos = nx.circular_layout(G)
    draw_graph(G, pos, "./figures/circular.pgf")

def gen_spring():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.spring_layout(G)
    draw_graph(G, pos, "./figures/spring.pgf")

def gen_arf():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.arf_layout(G)
    draw_graph(G, pos, "./figures/arf.pgf")

def gen_forceatlas2():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.forceatlas2_layout(G)
    draw_graph(G, pos, "./figures/forceatlas2.pgf")

def gen_kamada_kawai():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.kamada_kawai_layout(G)
    draw_graph(G, pos, "./figures/kamada_kawai.pgf")

def generate_random_planar_graph(n, m):
    seed = 69
    while True:
        G = nx.gnm_random_graph(n, m, seed=seed)
        seed += 1
        is_planar, _ = nx.check_planarity(G)
        if is_planar:
            return G

def gen_planar():
    G = generate_random_planar_graph(10, 15)
    pos = nx.planar_layout(G)
    draw_graph(G, pos, "./figures/planar.pgf")

def gen_shell():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.shell_layout(G)
    draw_graph(G, pos, "./figures/shell.pgf")

def gen_spectral():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.spectral_layout(G)
    draw_graph(G, pos, "./figures/spectral.pgf")

def gen_spiral():
    G = nx.Graph()
    G.add_edges_from([
        (1, 2), (1, 3), (2, 4), (2, 5), (2, 6),
        (3, 5), (3, 6), (4, 7), (4, 8), (5, 7),
        (5, 8), (6, 7), (6, 8), (7, 9), (8, 9)
    ])
    pos = nx.spiral_layout(G)
    draw_graph(G, pos, "./figures/spiral.pgf")

gen_fig1()
gen_bipartite()
gen_bfs()
gen_circular()
gen_spring()
gen_arf()
gen_forceatlas2()
gen_kamada_kawai()
gen_planar()
gen_shell()
gen_spectral()
gen_spiral()
