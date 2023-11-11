#%%
import networkx as nx
from networkx.drawing.nx_agraph import read_dot
from glob import glob

def _find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = _find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def get_graph(file_path):
    # Load the .gv file into a NetworkX graph
    graph = read_dot(glob(file_path)[0])
    return graph

def find_all_paths(file_path, start_node, end_node):
    graph = get_graph(file_path)
    # Convert the graph into a dictionary for easier traversal
    graph_dict = nx.to_dict_of_lists(graph)

    all_paths = _find_all_paths(graph_dict, start_node, end_node)

    return all_paths

def get_edge_values(file_path, paths):
    graph = get_graph(file_path)
    paths_with_values = []
    for path in paths:
        path_with_values = []
        for i in range(len(path) - 1):
            edge_value = graph[path[i]][path[i+1]].get('penwidth', None)
            path_with_values.append((path[i], path[i+1], edge_value))
        paths_with_values.append(path_with_values)
    return paths_with_values
#%%

# %%
