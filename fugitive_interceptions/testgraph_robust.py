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


def test_graph_func():
    G = nx.Graph()
    G.add_nodes_from([(0,0), (1,1), (2,1), (3,1), (4,1), (5,1), (1,0), (2,0), (3,0), (4,0), (5,0)])
    edge_list = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,7), (7,8), (8,9), (9,10), (10,11), (6,11)]
    G.add_edges_from([((0,0),(1,1)), ((1,1),(2,1)), ((2,1),(3,1)), ((3,1),(4,1)), ((4,1),(5,1)),
                            ((0,0),(1,0)), ((1,0),(2,0)), ((2,0),(3,0)), ((3,0),(4,0)), ((4,0),(5,0)), ((5,0),(5,1))])
    pos_list = [(0,0), (1,1), (2,1), (3,1), (4,1), (5,1), (1,0), (2,0), (3,0), (4,0), (5,0)]
    pos = {node: node for _, node in enumerate(G.nodes())}
    labels = {node: i for i, node in enumerate(G.nodes())}
    return G, labels, pos

T = 6
U = 1
R = 10
num_sensors = 1

graph, labels, pos = test_graph_func()
units_start = [(5, 0)]
fugitive_start = (0, 0)
sensor_locations = [(3, 1)]

#nx.draw(graph, pos=pos, labels=labels)

model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
                             num_sensors=num_sensors, sensor_locations=sensor_locations)

algorithm = PTreeOpt(model.f,
                     feature_bounds=[[0, T]] + [[0, 1]] * num_sensors,  # indicators
                     feature_names=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],  # indicator names
                     discrete_features=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],
                     discrete_actions=True,
                     action_names=[f"unit{u}_to_node{i}" for u in range(U) for i, _ in enumerate(graph.nodes)],
                     mu=20,  # 20
                     cx_prob=0.70,
                     population_size=2000,        #100
                     max_depth=10  # 5
                     )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    best_solution, best_score, snapshots = algorithm.run(max_nfe=50000,
                                                         log_frequency=100,
                                                         snapshot_frequency=100)

    pickle.dump(snapshots, open('results/snapshots.pkl', 'wb'))

    P = snapshots['best_P'][0]  #best policy tree
    colors = {f"unit{u}_to_node{i}": 'lightgrey' for u in range(U) for i, _ in enumerate(graph.nodes)}
    graphviz_export(P, 'figs/optimaltree.png', colordict=colors)  # creates one SVG

    model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
                             num_sensors=num_sensors, sensor_locations=sensor_locations)

    results_df, success = model.f(P, mode='simulation')
    print('Simulation: interception percentage: ', (sum(success.values()) * 100)/R)
    print(results_df['policy'])

    plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels)

