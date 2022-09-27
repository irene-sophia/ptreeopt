import sys

sys.path.append('../..')

# import pygraphviz as pgv
import numpy as np
import networkx as nx
import pickle
import logging
from random import sample
import random

from fugitive_interceptions.interception_model_interceptattarget import FugitiveInterception
from fugitive_interceptions.visualization import plot_result

from ptreeopt import PTreeOpt
from ptreeopt.plotting import *

def test_graph_func():
    G = nx.Graph()
    G.add_nodes_from([(0,1), (1,2), (2,2), (3,2), (4,2), (1,0), (2,0), (3,0), (4,0), (5,1)])
    G.add_edges_from([((0,1),(1,2)), ((1,2),(2,2)), ((2,2),(3,2)), ((3,2), (4,2)),
                      ((0,1),(1,0)), ((1,0),(2,0)), ((2,0),(3,0)), ((3,0),(4,0)),
                      ((4,2), (5,1)), ((4,0), (5,1)),
                      ((3,2), (3,0)), ((3,0), (3,2))])
    pos = {node: node for _, node in enumerate(G.nodes())}
    labels = {node: i for i, node in enumerate(G.nodes())}
    return G, labels, pos

N = 5
T = 4
U = 1
R = 10
num_sensors = 1

multiobj = False

graph, labels, pos = test_graph_func()
units_start = [(5,1)]
fugitive_start = (0,1)
sensor_locations = [(1,2)]
model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
                             num_sensors=num_sensors, sensor_locations=sensor_locations, multiobj=multiobj)

algorithm = PTreeOpt(model.f,
                     feature_bounds=[[0, T]] + [[0, 1]] * num_sensors,  # indicators
                     feature_names=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],  # indicator names
                     discrete_features=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],
                     discrete_actions=True,
                     action_names=[f"unit{u}_to_node{i}" for u in range(U) for i, _ in enumerate(graph.nodes)],
                     mu=20,  # 20
                     cx_prob=0.70,
                     population_size=100,
                     max_depth=1  # 5
                     , multiobj=multiobj
                     )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    best_solution, best_score, snapshots = algorithm.run(max_nfe=5000,
                                                         log_frequency=100,
                                                         snapshot_frequency=100)

    # best solution is not outputted AND logging frequency is strange - look into opt.py
    if multiobj:
        best_solution = best_solution[0]
    pickle.dump(snapshots, open('results/snapshots.pkl', 'wb'))

    # P = snapshots['best_P'][-1]  #best policy tree
    colors = {f"unit{u}_to_node{i}": 'lightgrey' for u in range(U) for i, _ in enumerate(graph.nodes)}
    graphviz_export(best_solution, 'figs/optimaltree.png', colordict=colors)  # creates one SVG

    model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
                                 num_sensors=num_sensors, sensor_locations=sensor_locations, multiobj=multiobj)

    results_df, success = model.f(best_solution, mode='simulation')
    print('Simulation: interception succeeded: ', success)
    print(results_df['policy'])

    plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels)

