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
            try:
                connections = value["connection"]
            except KeyError:
                print("t")
            if key not in _dict.keys():
                _dict[key] = {}
            if connections:
                for _key, _value in connections.items():
                    out = _key.split("<->")
                    if out[1] not in _dict.keys():
                        _dict[out[1]] = {}
                    logging.debug(f"key: {_key}\n\t Route: {_value['route']}\n\t Reversed: {_value['route'][::-1]}")

                    is_chariot_route = False
                    if 'chariot' in self.node_definitions['Nodes'][out[1]].keys() and 'chariot' in self.node_definitions['Nodes'][out[0]].keys():
                        is_chariot_route = True

                    if out[1] not in _dict[key]:
                        dist = total_dist(_value['route']) if not is_chariot_route else 0
                        _dict[key][out[1]] = {"route": _value['route'], "cost": dist}
                    if out[0] not in _dict[out[1]]:
                        dist = total_dist(_value['route']) if not is_chariot_route else 0
                        _dict[out[1]][out[0]] = {"route": _value['route'][::-1], "cost": dist}
        return _dict

    def get_best_path_from_coordinates(self, start, end):
        print(f"Start Coordinate: {start}, End waypoint: {end}")
        start_waypoint = self.get_closest_waypoint_to_coordinate(start)
        end_waypoint = self.get_closest_waypoint_to_coordinate(end, safe_start_node=False)
        print(f"Start Waypoint: {start_waypoint}, End Waypoint: {end_waypoint}")

        distances = self.dijkstra(self.graph, start_waypoint)
        path = self.find_path(distances, start_waypoint, end_waypoint)
        out = self.resolve_coordinates(self.graph, path)
        extended = self.append_closest_connection(distances, out, start_waypoint, start)
        return extended

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

    def append_closest_connection(self, distances, path, node, start_coord):
        if node not in self.graph:
            return path

        least_dist = float("inf")
        closest_path = ""
        closest_point = None

        # Get closest connecting path to node
        for idx, val in enumerate(distances[node]['nodes']):
            paths = self.graph[node][val]['route']
            for _idx, _val in enumerate(paths):
                dist = manhattan(np.asarray(_val), np.asarray(start_coord))
                if dist < least_dist:
                    closest_path = val
                    closest_point = _val
                    least_dist = dist

        additional = []
        save_points = False
        if not closest_point:
            return path

        # add points starting at closest to the node connection
        for idx, val in enumerate(self.graph[node][closest_path]['route'][::-1]):
            if save_points:
                additional.append(val)
            if closest_point == val:
                additional.append(val)
                save_points = True

        additional.extend(path)

        return additional

    def resolve_coordinates(self, graph, path):
        coordinates = []
        for idx, val in enumerate(path):
            if idx < len(path) - 1:
                curr_is_chariot = self.node_definitions['Nodes'][path[idx]].get('chariot')
                next_is_chariot = self.node_definitions['Nodes'][path[idx+1]].get('chariot')
                if curr_is_chariot and next_is_chariot:
                    coordinates.append(graph[val][path[idx + 1]]['route'][0])
                    coordinates.append(f"destination_{path[idx+1]}")
                    coordinates.append(graph[val][path[idx + 1]]['route'][1])
                else:
                    coordinates.extend(graph[val][path[idx + 1]]['route'])
        if len(coordinates) == 0:
            coordinates.append(self.node_definitions['Nodes'][path[0]]["coordinate"])
        return coordinates
