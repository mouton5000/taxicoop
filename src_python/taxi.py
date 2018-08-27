import copy
from typing import Callable, List

from utils import travel_time


class Taxi:

    def __init__(self, request, capacity: int=2, speed: int=40):
        """
        list of request indices in the same order as the taxi drives them.
        positive indices stand for PU points when negative indices stand for DO points.
        """
        B_source = request.PU_datetime[0]
        B_destination = request.PU_datetime[0] + travel_time(request.PU_coordinates, request.DO_coordinates)
        # e.g. [(B3_source, request_3, PU_coordinates_3), (B6_source, request_6, PU_coordinates_6),
        #       (B3_destination, request_3, DO_coordinates_3), (B6_destination, request_6, DO_coordinates_6)]
        self.route = [[B_source, request, request.PU_coordinates], [B_destination, request, request.DO_coordinates]]
        self.capacity = capacity
        self.speed = speed

    def insert(self, request: Callable, method: str="IA"):
        """
        IA stands for the exhaustive insertion method
        IB stands for the heuristic insertion method
        """
        if any(r[1].id == request.id for r in self.route):
            raise Exception("Impossible to insert a request in a route where it is already there")
        if method is "IA":
            for pu in range(len(self.route) + 1):
                route1 = copy.deepcopy(self.route)
                route1.insert(pu, [request.PU_datetime[0], request, request.PU_coordinates])
                for do in range(pu + 1, len(route1) + 1):
                    route2 = copy.deepcopy(route1)
                    route2.insert(do, [request.PU_datetime[0] + travel_time(request.PU_coordinates, request.DO_coordinates),
                                request, request.DO_coordinates])
                    valid = self.is_valid(route2)
                    self.update_delivery_datetimes(route2)
                    valid *= self.is_valid(route2)
                    if valid:
                        self.route = route2
                        return
            raise Exception("could not insert request %d into the route %s" % (request.id, self.route))

        if method is "IB":
            """
            TODO
            """

    def remove(self, request: int):
        """
        """
        self.route = [req for req in self.route if req[1].id != request]

    def swap(self, pos1: int, pos2: int):
        """
        """
        route = copy.deepcopy(self.route)
        route[pos1], route[pos2] = route[pos2], route[pos1]
        served_requests = []
        for i in range(len(route)):
            # source point
            if route[i][1].id not in served_requests:
                route[i][0] = route[i][1].PU_datetime[0]
            
            # destination point
            else:
                route[i][0] = route[i][1].DO_datetime[0]
            served_requests.append(route[i][1].id)

        try:  
            self.is_valid(route)
            self.update_delivery_datetimes(route)
            self.is_valid(route)
            self.route = route
        except Exception:
            raise Exception("could not swap point %d and point %d in route %s" % (pos1, pos2, self.route))
    
    @property
    def delay(self):
        request_delays = {}
        for point in self.route:
            if point[1].id not in request_delays:
                request_delays[point[1].id] = point[0]
            else:
                request_delays[point[1].id] = abs(request_delays[point[1].id] - point[0]) - travel_time(point[1].PU_coordinates, point[1].DO_coordinates)
        assert len(request_delays) ==  len(self.route) / 2
        delay = round(sum(request_delays.values()), 3)
        assert delay >= 0
        return delay

    def get_loading_timeline(self, route) -> List[int]:
        served_request = []
        loading_timeline = [0]
        for x in route:
            if x[1].id not in served_request:
                loading_timeline.append(loading_timeline[-1] + 1)
                served_request.append(x[1].id)
            else:
                loading_timeline.append(loading_timeline[-1] - 1)

        # for future tests
        assert len(loading_timeline) == len(route) + 1
        assert loading_timeline[-1] == 0

        return loading_timeline

    def is_valid(self, route) -> bool:
        """
        check:
        1. the capacity constraint
        2. no stop in a taxi's route
        3. the time windows of the passengers
        """

        # check the capacity constraint
        loading_timeline = self.get_loading_timeline(route)
        if max(loading_timeline) > self.capacity:
            #print("The maximum number of passengers served at the same time %d"
            #                "exceed the taxi capacity %d" % (max(loading_timeline, self.capacity)))
            return False

        # no stop in a taxi's route
        if 0 in loading_timeline[1:-1]:
            #print("The taxi has to stop after droping the last client in the vehicle.")
            return False

        # check the time windows of the passengers
        served_requests = []
        for i in range(len(route)):

            # source point
            if route[i][1].id not in served_requests and route[i][0] > route[i][1].PU_datetime[1]:
                #print("The taxi can not pick up the client %d after the end of the time window %d"
                 #               % (route[i][1].id, route[i][1].PU_datetime[1]))
                return False
            
            # destination point
            elif route[i][1].id in served_requests and route[i][0] > route[i][1].DO_datetime[1]:
                #print("The taxi can not drop off the client %d after the end of the time window %d"
                #                % (route[i][1].id, route[i][1].DO_datetime[1]))
                return False
            served_requests.append(route[i][1].id)
        return True

    def update_delivery_datetimes(self, route):
        for i in range(len(route)-1):
            time_to_next_point = travel_time(route[i][2], route[i+1][2])
            if route[i][0] + time_to_next_point > route[i+1][0]:
                route[i+1][0] = route[i][0] + time_to_next_point
            
            # source point
            #if self.route[i][1] not in served_requests:
            #    if self.route[i][0] < self.route[i][1].PU_datetime[0]:
            #        self.route[i][0] = self.route[i][1].PU_datetime[0]
            # destination point
            #else:
            #    if self.route[i][0] < self.route[i][1].DO_datetime[0]:
            #        self.route[i][0] = self.route[i][1].DO_datetime[0]
