import heapq
import logging
from collections import deque
from queue import Queue

import numpy as np
import pprint
import yaml

from AtitdScripts.utils import manhattan, total_dist


class WebWalkerTree(object):
    """

    """

    def __init__(self, node_definitions):
        with open(node_definitions, 'r') as f:
            self.node_definitions = yaml.safe_load(f)
        self.graph = self.generate_graph({})
        self.min_dist_graph = {}

    def generate_graph(self, _dict):
        assert self.node_definitions is not None
        assert "Nodes" in self.node_definitions.keys()

        for key, value in self.node_definitions['Nodes'].items():
            connections = value["connection"]
            if key not in _dict.keys():
                _dict[key] = {}
            if connections:
                for _key, _value in connections.items():
                    out = _key.split("<->")
                    if out[1] not in _dict.keys():
                        _dict[out[1]] = {}
                    logging.debug(f"key: {_key}\n\t Route: {_value['route']}\n\t Reversed: {_value['route'][::-1]}")
                    if out[1] not in _dict[key]:
                        _dict[key][out[1]] = {"route": _value['route'],
                                              "cost": total_dist(_value['route'])}
                    if out[0] not in _dict[out[1]]:
                        _dict[out[1]][out[0]] = {"route": _value['route'][::-1],
                                                 "cost": total_dist(_value['route'])}
        return _dict

    def get_best_path_from_coordinates(self, start, end):
        start_waypoint = self.get_closest_waypoint_to_coordinate(start)
        end_waypoint = self.get_closest_waypoint_to_coordinate(end, safe_start_node=False)

        distances = self.dijkstra(self.graph, start_waypoint)
        path = self.find_path(distances, start_waypoint, end_waypoint)
        out = self.resolve_coordinates(self.graph, path)
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

    @staticmethod
    def dijkstra(graph, starting_vertex):
        min_dist_graph = {}
        distances = {vertex: float('inf') for vertex in graph}
        distances[starting_vertex] = 0
        pq = [(0, 0, starting_vertex, None)]

        while len(pq) > 0:
            current_distance, edge_weight, current_vertex, parent = heapq.heappop(pq)
            if current_distance > distances[current_vertex]:
                continue

            if parent:
                if parent not in min_dist_graph.keys():
                    min_dist_graph[parent] = {"cost": float('inf'), "nodes": []}
                if current_distance < min_dist_graph[parent]["cost"]:
                    min_dist_graph[parent]["cost"] = current_distance
                min_dist_graph[parent]["nodes"].append(current_vertex)

            for neighbor, weight in graph[current_vertex].items():
                distance = current_distance + weight['cost']
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, weight['cost'], neighbor, current_vertex))

        return min_dist_graph

    def find_path(self, graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in graph.keys():
            return None
        for node in graph[start]['nodes']:
            if node not in path:
                newpath = self.find_path(graph, node, end, path)
                if newpath:
                    return newpath
        return None

    def resolve_coordinates(self, graph, path):
        coorindates = []
        for idx, val in enumerate(path):
            if idx < len(path) - 1:
                coorindates.extend(graph[val][path[idx + 1]]['route'])
        if len(coorindates) == 0:
            coorindates.append(self.node_definitions['Nodes'][path[0]]["coordinate"])
        return coorindates
