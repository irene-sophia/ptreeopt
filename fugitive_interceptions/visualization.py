import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import imageio
import glob
import os


def draw_edges(graph, results_df, step, T, U):
    fugitive_route = results_df['fugitive_route'].tolist()
    edges_fugitive = []
    edges_units = []

    if step in range(1,T):
        # for i in range(1, step+1):  #to plot entire tail
        edges_fugitive1 = [(fugitive_route[step], fugitive_route[step-1])]
        edges_fugitive2 = [(fugitive_route[step-1], fugitive_route[step])]
        edges_fugitive.extend(tuple(edges_fugitive1))
        edges_fugitive.extend(tuple(edges_fugitive2))

        for u in range(U):
            unit_route = results_df['unit'+str(u)]
            edges_units.extend([(unit_route[step], unit_route[step-1])])
            edges_units.extend([(unit_route[step-1], unit_route[step])])

    edge_colormap = ['lightgray'] * len(graph.edges())
    edge_weightmap = [1] * len(graph.edges())
    for index, edge in enumerate(graph.edges()):
        # fugitive
        if edge in edges_fugitive:
            edge_colormap[index] = 'tab:orange'
            edge_weightmap[index] = 2
        # units
        if edge in edges_units:
            edge_colormap[index] = 'tab:blue'
            edge_weightmap[index] = 2

    return edge_colormap, edge_weightmap


def draw_nodes(graph, results_df, sensor_locations, U, step, labels, pos):
    node_colormap = ['lightgray']*len(graph.nodes)
    labels_at_step = labels.copy()

    sameplace_colormap = []
    sameplace_pos = dict()
    sameplace_graph = nx.Graph()

    for index, node in enumerate(graph.nodes()):
        # sensors
        if node in sensor_locations:
            node_colormap[index] = 'tab:red'
            labels_at_step[node] = 'sensor'+str(sensor_locations.index(node))
        # fugitive route
        if node == results_df['fugitive_route'].tolist()[step]:
            if node_colormap[index] == 'lightgray':
                node_colormap[index] = 'tab:orange'
            else:
                sameplace_graph.add_node(node)
                sameplace_pos[node] = pos[node]
                sameplace_colormap.append('tab:orange')
        # unit routes final
        for u in range(U):
            if node == results_df['unit'+str(u)].tolist()[step]:
                if node_colormap[index] == 'lightgray':
                    node_colormap[index] = 'tab:blue'
                    labels_at_step[node] = 'unit'+str(u)
                else:
                    sameplace_graph.add_node(node)
                    sameplace_pos[node] = pos[node]
                    sameplace_colormap.append('tab:blue')

    return node_colormap, labels_at_step, sameplace_colormap, sameplace_graph, sameplace_pos


def plot_result(graph, pos, T, U, results_df, sensor_locations, labels):
    for step in range(T):
        node_colormap, labels_at_step, sameplace_colormap, sameplace_graph, sameplace_pos = draw_nodes(graph, results_df, sensor_locations, U, step, labels, pos)
        edge_colormap, edge_weightmap = draw_edges(graph, results_df, step, T, U)

        nx.draw_networkx_nodes(G=graph, pos=pos,
                               node_color=node_colormap, alpha=0.9,
                               node_shape=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'))

        nx.draw_networkx_labels(G=graph, pos=pos, labels=labels_at_step, font_size=10)

        if len(sameplace_pos) > 0:
            nx.draw_networkx_nodes(G=sameplace_graph, pos=sameplace_pos,
                                   node_color=sameplace_colormap, alpha=0.9,
                                   node_shape=matplotlib.markers.MarkerStyle(marker='o', fillstyle='bottom'))

        nx.draw_networkx_edges(G=graph, pos=pos, edge_color=edge_colormap, width=edge_weightmap)

        labels_at_step.clear()
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        #plt.legend(loc=1)
        ax.text(0.3, 0.3, 't = '+str(step), bbox=dict(facecolor='yellow', alpha=0.5))
        plt.savefig('figs/test_' + str(step) + '.png')
        plt.clf()

    with imageio.get_writer('figs/mygif.gif', mode='I', duration=1) as writer:
        for filename in ['figs/test_'+str(step)+'.png' for step in range(T)]:
            image = imageio.imread(filename)
            writer.append_data(image)

    # remove files
    for filename in set(['figs/test_'+str(step)+'.png' for step in range(T)]):
        os.remove(filename)
