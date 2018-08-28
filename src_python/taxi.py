import copy
from typing import Callable, List

from utils import travel_time


# types of insertion methods
IA = "IA" # exhaustive method
IB = "IB" # heuristic method

class Taxi:

    def __init__(self, request, capacity: int=2, speed: int=40):
        """
        list of points and corresponding requests in the route of a Taxi in the same order as the taxi travels through them.
        Each point in self.route is reprensented by (time, request, coordinates).
        Example of self.route 2 requests served :
        [(B3_source, request_3, PU_coordinates_3), (B6_source, request_6, PU_coordinates_6),
        (B3_destination, request_3, DO_coordinates_3), (B6_destination, request_6, DO_coordinates_6)]
        """
        B_source = request.PU_datetime[0]
        B_destination = request.PU_datetime[0] + travel_time(request.PU_coordinates, request.DO_coordinates)
        self.route = [[B_source, request, request.PU_coordinates], [B_destination, request, request.DO_coordinates]]
        self.capacity = capacity
        self.speed = speed

    def insert(self, request: Callable, method: str):
        """
        Insertion methods describe in Santos 2015.
        IA stands for the exhaustive  method.
        IB stands for the heuristic method.
        """
        if any(r[1].id == request.id for r in self.route):
            raise Exception("Impossible to insert a request in a route where it is already there")
        
        if method is IA:
            for pu in range(len(self.route) + 1):
                route1 = copy.deepcopy(self.route)
                route1.insert(pu, [request.PU_datetime[0], request, request.PU_coordinates])
                for do in range(pu + 1, len(route1) + 1):
                    route2 = copy.deepcopy(route1)
                    route2.insert(do, [request.PU_datetime[0] + travel_time(request.PU_coordinates, request.DO_coordinates),
                                  request, request.DO_coordinates])
                    valid = self.is_valid(route2)
                    if valid:
                        self.route = route2
                        return
            raise Exception("could not insert request %d into the route %s" % (request.id, self.route))

        if method is IB:
            cur_delay = 100000
            for pu in range(len(self.route) + 1):
                route1 = copy.deepcopy(self.route)
                route1.insert(pu, [request.PU_datetime[0], request, request.PU_coordinates])
                for do in range(pu + 1, len(route1) + 1):
                    route2 = copy.deepcopy(route1)
                    route2.insert(do, [request.PU_datetime[0] + travel_time(request.PU_coordinates, request.DO_coordinates),
                                  request, request.DO_coordinates])
                    valid = self.is_valid(route2)
                    if valid and self.delay(route2) < cur_delay:
                        cur_delay = self.delay(route2)
                        route3 = route2
            try:
                self.route = route3
                return
            except UnboundLocalError:
                raise Exception("could not insert request %d into the route %s" % (request.id, self.route))

    def remove(self, request: int):
        """
        Removes the input request from the route.
        """
        self.route = [req for req in self.route if req[1].id != request]

    def swap(self, pos1: int, pos2: int):
        """
        Function to swap points in pos1 and pos2 in the route if valid.
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
    
    def delay(self, route):
        """
        Returns the delay of the route.
        """
        request_delays = {}
        for point in route:
            if point[1].id not in request_delays:
                request_delays[point[1].id] = point[0]
            else:
                request_delays[point[1].id] = abs(request_delays[point[1].id] - point[0]) - travel_time(point[1].PU_coordinates, point[1].DO_coordinates)
        assert len(request_delays) ==  len(route) / 2
        delay = round(sum(request_delays.values()), 3)
        assert delay >= 0
        return delay

    def get_loading_timeline(self, route) -> List[int]:
        """
        Returns the loading timeline of the taxi through its travel.
        """
        served_request = []
        loading_timeline = [0]
        for x in route:
            if x[1].id not in served_request:
                loading_timeline.append(loading_timeline[-1] + 1)
                served_request.append(x[1].id)
            else:
                loading_timeline.append(loading_timeline[-1] - 1)

        assert len(loading_timeline) == len(route) + 1
        assert loading_timeline[-1] == 0
        return loading_timeline

    def is_valid(self, route) -> bool:
        """
        Checks the validity of the route according to the following constraints :
            1. The capacity constraint
            2. No stops permitted in the taxi's route
            3. The time windows of the passengers
        """
        served_request = []
        loading_timeline = [0]
        for i, x in enumerate(route):
            if x[1].id not in served_request:
                loading_timeline.append(loading_timeline[-1] + 1)
                # 1. The capacity constraint
                if loading_timeline[-1] > self.capacity:
                    return False
                served_request.append(x[1].id)
            else:
                loading_timeline.append(loading_timeline[-1] - 1)
                # 2. No stops permitted in the taxi's route
                if loading_timeline[-1] == 0 and i < len(route) - 1:
                    return False

        # sanity checks
        assert len(loading_timeline) == len(route) + 1
        assert loading_timeline[-1] == 0

        # The time windows of the passengers
        served_requests = []
        for i in range(len(route) - 1):
            # updates the delivery datetimes of the route after the insertions
            time_to_next_point = travel_time(route[i][2], route[i+1][2])
            if route[i][0] + time_to_next_point > route[i+1][0]:
                route[i+1][0] = route[i][0] + time_to_next_point

            # source point
            if route[i][1].id not in served_requests and route[i][0] > route[i][1].PU_datetime[1]:
                return False
            # destination point
            elif route[i][1].id in served_requests and route[i][0] > route[i][1].DO_datetime[1]:
                return False
            served_requests.append(route[i][1].id)

        # last point of the route
        if route[-1][0] > route[-1][1].DO_datetime[1]:
            return False
        #print("True")
        return True
