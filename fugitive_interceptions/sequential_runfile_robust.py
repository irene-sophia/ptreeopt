import sys

sys.path.append('..')

# import pygraphviz as pgv
import numpy as np
import networkx as nx
import pickle
import logging
from random import sample
import random

from interception_model_robust import FugitiveInterception
from visualization import plot_result

from ptreeopt import PTreeOpt
from ptreeopt.plotting import *


# Example to run optimization and save results
#np.random.seed(112)


def graph_func(N):
    """
    Generate manhattan graph

    Parameters
    ----------
    N : int
        width of graph.

    Returns
    -------
    G : networkx graph
        graph of size (N,N).
    labels : dict
        {pos: vertex number}.
    pos : dict
        {vertex:pos}.

    """

    G = nx.grid_2d_graph(N, N)
    pos = dict((n, n) for n in G.nodes())
    #labels = dict(((i, j), i * N + j) for i, j in G.nodes())  #
    labels = {node: str(i) for i, node in enumerate(G.nodes())}

    return G, labels, pos

N = 10
T = 10
U = 3
R = 20
num_sensors = 3

graph, labels, pos = graph_func(N)
units_start = sample(list(graph.nodes()), U)
fugitive_start = sample(list(graph.nodes()), 1)[0]
sensor_locations = sample(list(graph.nodes()), num_sensors)
model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
                             num_sensors=num_sensors, sensor_locations=sensor_locations)

algorithm = PTreeOpt(model.f,
                     feature_bounds=[[0, T]] + [[0, 1.5]] * num_sensors,  # indicators
                     feature_names=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],  # indicator names
                     discrete_features=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],
                     discrete_actions=True,
                     action_names=[f"unit{u}_to_node{i}" for u in range(U) for i, _ in enumerate(graph.nodes)],
                     mu=40,  # 20
                     cx_prob=0.70,
                     population_size=100,
                     max_depth=10  # 5
                     )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    best_solution, best_score, snapshots = algorithm.run(max_nfe=10000,
                                                         log_frequency=100,
                                                         snapshot_frequency=100)

    pickle.dump(snapshots, open('results/snapshots.pkl', 'wb'))

    # P = snapshots['best_P'][-1]  #best policy tree
    colors = {f"unit{u}_to_node{i}": 'lightgrey' for u in range(U) for i, _ in enumerate(graph.nodes)}
    graphviz_export(best_solution, 'figs/optimaltree.png', colordict=colors)  # creates one SVG

    model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
                             num_sensors=num_sensors, sensor_locations=sensor_locations)

    results_df, success = model.f(best_solution, mode='simulation')
    print('Simulation: interception percentage: ', (sum(success.values()) * 100)/R)
    print(results_df['policy'])

    plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels)

