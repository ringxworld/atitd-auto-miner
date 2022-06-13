import logging
import numpy as np
import pprint
import yaml

from AtitdScripts.utils import manhattan


class WebWalkerTree(object):
    """

    """

    def __init__(self, node_definitions):
        with open(node_definitions, 'r') as f:
            self.node_definitions = yaml.safe_load(f)
        self.graph = self.generate_graph({})

    def generate_graph(self, _dict):
        assert self.node_definitions is not None
        assert "Nodes" in self.node_definitions.keys()

        for key, value in self.node_definitions['Nodes'].items():
            connections = value["connection"]
            if key not in _dict.keys():
                _dict[key] = []
            if connections:
                for _key, _value in connections.items():
                    out = _key.split("<->")
                    if out[1] not in _dict.keys():
                        _dict[out[1]] = []
                    logging.debug(f"key: {_key}\n\t Route: {_value['route']}\n\t Reversed: {_value['route'][::-1]}")
                    if out[1] not in _dict[key]:
                        _dict[key].append({out[1]: _value['route']})
                    if out[0] not in _dict[out[1]]:
                        _dict[out[1]].append({out[0]: _value['route'][::-1]})
        pprint.pprint(_dict)

        return _dict

    def get_best_path_from_coordinates(self, start, end):
        start_waypoint = self.get_closest_waypoint_to_coordinate(start)
        end_waypoint = self.get_closest_waypoint_to_coordinate(end, safe_start_node=False)

        out = self.resolve_coordinates(self.graph, self.find_path(self.graph, start_waypoint, end_waypoint))
        return out

    def get_closest_waypoint_to_coordinate(self, coordinate, safe_start_node=True):
        min = float('inf')
        closest_waypoint = None
        for key, value in self.node_definitions["Nodes"].items():
            if safe_start_node and 'rough_terrain' in value:
                continue
            dist = manhattan(np.asarray(value["coordinate"]), np.asarray(coordinate))
            if dist < min:
                min = dist
                closest_waypoint = key

        return closest_waypoint

    def find_path(self, graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in graph.keys():
            return None
        for node in graph[start]:
            node_key = list(node.keys())[0]
            if node_key not in path:
                newpath = self.find_path(graph, node_key, end, path)
                if newpath:
                    return newpath
        return None

    def resolve_coordinates(self, graph, path):
        coorindates = []
        for idx, val in enumerate(path):
            if idx + 1 < len(path):
                paths = graph[val]
                for i in paths:
                    if path[idx + 1] in i.keys():
                        coorindates.extend(i[path[idx + 1]])
        if len(coorindates) == 0:
            coorindates.append(self.node_definitions['Nodes'][path[0]]["coordinate"])
        return coorindates

