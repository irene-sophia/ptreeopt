import sys

sys.path.append('..')

# import pygraphviz as pgv
import numpy as np
import networkx as nx
import pickle
import logging
from random import sample
import random

from fugitive_interceptions.interception_model_sorted_filtered import FugitiveInterception
from fugitive_interceptions.visualization import plot_result
# from fugitive_interceptions.networks import rotterdam_graph as graph_func
from fugitive_interceptions.networks import manhattan_graph as graph_func

from ptreeopt import PTreeOpt
from ptreeopt.plotting import *


def unit_nodes(start_police, graph, run_length, U):
    unit_nodes = {}
    for u in range(U):
        unit_nodes[u] = list(nx.single_source_shortest_path_length(G=graph, source=start_police[u], cutoff=run_length).keys())

    return unit_nodes


def sort_and_filter_nodes(graph, start_fugitive, route_fugitive, unit_nodes, U):
    """
    Sorts the nodes of a graph on distance to start_fugitive,
    filters the nodes on reachability by the fugitive,
    and returns the updated labels
    """

    distance_dict = {}
    for node in graph.nodes():
        distance_dict[node] = nx.shortest_path_length(G=graph,
                                                      source=start_fugitive,
                                                      target=node,
                                                      weight='travel_time',
                                                      method='dijkstra')

    nx.set_node_attributes(graph, distance_dict, "distance")

    fugitive_nodes = set([y for x in [*route_fugitive] for y in x])

    nodesdict_perunit_sorted = {}
    nodesdict_perunit_inv_sorted = {}
    actions = []

    for u in range(U):
        police_nodes = set([y for x in [unit_nodes[u]] for y in x])

        node_subset = (list(fugitive_nodes.intersection(police_nodes)))
        subgraph = graph.subgraph(nodes=node_subset)

        nodesdict_perunit_sorted['unit'+str(u)] = {node: 'node' + str(index) for index, node in enumerate(sorted(subgraph.nodes(), key=lambda n: subgraph.nodes[n]['distance']))}
        nodesdict_perunit_inv_sorted['unit'+str(u)] = {'node' + str(index): node for index, node in enumerate(sorted(subgraph.nodes(), key=lambda n: subgraph.nodes[n]['distance']))}

        actions_unit = [f"unit{u}_to_node{i}" for i, node in enumerate(sorted(subgraph.nodes(), key=lambda n: subgraph.nodes[n]['distance']))]
        actions.extend(actions_unit)

    return nodesdict_perunit_sorted, nodesdict_perunit_inv_sorted, actions


def escape_route(graph, fugitive_start, T):
    walk = []
    node = fugitive_start
    walk.append(node)

    for i in range(T - 1):
        list_neighbor = list(graph.neighbors(node))

        if i == 0:
            previous_node = node
            nextnode = random.choice(list_neighbor)

        else:
            # exclude previous node for 'normal' walk
            # don't do this for dead ends
            if len(list_neighbor) > 1:
                if previous_node in list_neighbor:
                    list_neighbor.remove(previous_node)

            # save previous node
            previous_node = node
            nextnode = random.choice(list_neighbor)

        walk.append(nextnode)
        node = nextnode

    return walk



if __name__ == '__main__':
    N = 10
    T = 10
    U = 3
    R = 20
    num_sensors = 3

    num_repetitions = 10
    num_seeds = 10

    for rep in [7]:
        graph, labels, pos = graph_func(N=N)  # change labels to pointers

        # open pickles from prev experiment to guarantee same starting configuration
        fugitive_start = pd.read_pickle(r'results/start_fug_rep{}_seed0.pkl'.format(rep))
        units_start = pd.read_pickle(r'results/start_units_rep{}_seed0.pkl'.format(rep))
        sensor_locations = pd.read_pickle(r'results/sensors_rep{}_seed0.pkl'.format(rep))
        fugitive_routes_db = pd.read_pickle(r'results/routes_fug_rep{}_seed0.pkl'.format(rep))
        #fugitive_routes_db = pd.read_pickle(r'results/routes_fug_100routes_rep{}_seed0.pkl'.format(rep))

        # simulate fugitive escape routes (if changing number of routes)
        # fugitive_routes_db = []
        # for r in range(R):
        #     route = escape_route(graph, fugitive_start, T)
        #     fugitive_routes_db.append(route)
        # pickle.dump(fugitive_routes_db, open('results/routes_fug_{}routes_rep{}_seed0.pkl'.format(R, rep), 'wb'))

        unit_nodes_data = unit_nodes(units_start, graph, T, U)
        nodesdict_perunit_sorted, nodesdict_perunit_inv_sorted, actions = sort_and_filter_nodes(graph, fugitive_start, fugitive_routes_db, unit_nodes_data, U)

        print('number of actions, rep ', rep, ': ', len(actions))

        for seed_rep in range(10):
            # save experiment parameters
            random.seed(seed_rep)
            np.random.seed(seed_rep)

            pickle.dump(sensor_locations, open('results/sorted_filtered/sensors_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))
            pickle.dump(units_start, open('results/sorted_filtered/start_units_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))
            pickle.dump(fugitive_start, open('results/sorted_filtered/start_fug_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))
            pickle.dump(fugitive_routes_db, open('results/sorted_filtered/routes_fug_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))

            model = FugitiveInterception(T, U, R, graph=graph,
                                         units_start=units_start, nodesdict_perunit_sorted=nodesdict_perunit_inv_sorted,
                                         fugitive_start=fugitive_start, fugitive_routes_db=fugitive_routes_db,
                                         num_sensors=num_sensors, sensor_locations=sensor_locations,
                                         seed=seed_rep)

            algorithm = PTreeOpt(model.f,
                                 feature_bounds=[[0, T]] + [[0.5, 1.5]] * num_sensors,  # indicators
                                 feature_names=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],  # indicator names
                                 discrete_features=['Minute'] + [f"sensor{s}" for s in range(num_sensors)],
                                 discrete_actions=True,
                                 action_names=actions,
                                 mu=20,  # 20
                                 cx_prob=0.70,
                                 population_size=100,
                                 max_depth=10  # 5
                                 )

            logging.basicConfig(level=logging.INFO,
                                format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

            best_solution, best_score, snapshots = algorithm.run(max_nfe=100000,
                                                                 log_frequency=100,
                                                                 snapshot_frequency=100)

            pickle.dump(snapshots, open('results/sorted_filtered/snapshots_rep{}_seed{}.pkl'.format(rep, seed_rep), 'wb'))

            # P = snapshots['best_P'][-1]  #best policy tree
            colors = {f"unit{u}_to_node{i}": 'lightgrey' for u in range(U) for i, _ in enumerate(graph.nodes)}
            graphviz_export(best_solution, 'figs/sorted_filtered/optimaltree_rep{}_seed{}.png'.format(rep, seed_rep), colordict=colors)  # creates one SVG

            # model = FugitiveInterception(T, U, R, graph=graph, units_start=units_start, fugitive_start=fugitive_start,
            #                          num_sensors=num_sensors, sensor_locations=sensor_locations)

            print('finished seed ', seed_rep+1, 'of repetition ', rep+1, '(experiment number ',
                  (rep*num_seeds)+seed_rep+1, 'of ', (num_seeds*num_repetitions), ').')
            # results_df, success = model.f(best_solution, mode='simulation')  # re-initializes! only use for visuals
            # print('Simulation: interception percentage: ', (sum(success.values()) * 100)/R)
            # print(results_df['policy'])

            # plot_result(graph, pos, T, U, R, results_df, success, sensor_locations, labels)