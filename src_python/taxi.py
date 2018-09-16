import copy
from typing import Callable

from collections import defaultdict
from haversine import haversine

from utils import travel_time


# types of insertion methods
IA = "IA"  # exhaustive method
IB = "IB"  # heuristic method


class Taxi:

    def __init__(self, request, capacity: int, speed: int):
        """
        list of points and corresponding requests in the route of a Taxi
            in the same order as the taxi travels through them.
        Each point in self.route is reprensented by (time, request, coordinates).
        Example of self.route 2 requests served :
        [(B3_source, request_3, PU_coordinates_3), (B6_source, request_6, PU_coordinates_6),
        (B3_destination, request_3, DO_coordinates_3), (B6_destination, request_6, DO_coordinates_6)]
        """
        B_source = request.PU_datetime[1]
        B_destination = B_source + travel_time(request.PU_coordinates, request.DO_coordinates)
        self.route = [[B_source, request, request.PU_coordinates], [B_destination, request, request.DO_coordinates]]
        self.capacity = capacity
        self.speed = speed

    def insert(self, request: Callable, alpha: float, method: str):
        """
        Insertion methods describe in Santos 2015.
        IA stands for the exhaustive  method.
        IB stands for the heuristic method.
        """
        if any(r[1].id == request.id for r in self.route):
            raise ValueError("Impossible to insert a request in a route where it is already there")

        if method is IA:
            for pu in range(len(self.route) + 1):
                route1 = copy.deepcopy(self.route)
                route1.insert(pu, [request.PU_datetime[1], request, request.PU_coordinates])
                for do in range(pu + 1, len(route1) + 1):
                    route2 = copy.deepcopy(route1)
                    route2.insert(do, [request.PU_datetime[1] + travel_time(request.PU_coordinates,
                                                                            request.DO_coordinates),
                                  request, request.DO_coordinates])
                    valid = self.is_valid(route=route2, alpha=alpha)
                    if valid:
                        self.route = route2
                        return
            raise StopIteration("could not insert request %d into the route %s" % (request.id, self.route))

        if method is IB:
            cur_delay = 100000
            for pu in range(len(self.route) + 1):
                route1 = copy.deepcopy(self.route)
                route1.insert(pu, [request.PU_datetime[1], request, request.PU_coordinates])
                for do in range(pu + 1, len(route1) + 1):
                    route2 = copy.deepcopy(route1)
                    route2.insert(do, [request.PU_datetime[1] + travel_time(request.PU_coordinates,
                                                                            request.DO_coordinates),
                                  request, request.DO_coordinates])
                    valid = self.is_valid(route=route2, alpha=alpha)
                    if valid and self.delay(route=route2) < cur_delay:
                        cur_delay = self.delay(route=route2)
                        route3 = route2
            try:
                self.route = route3
                return
            except UnboundLocalError:
                raise StopIteration("could not insert request %d into the route %s" % (request.id, self.route))

    def remove(self, request_id: int):
        """
        Removes the input request from the route.
        """
        self.route = [req for req in self.route if req[1].id != request_id]

    def swap(self, pos1: int, pos2: int, alpha: float):
        """
        Function currently not used.
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
            self.is_valid(route=route, alpha=alpha)
            self.update_delivery_datetimes(route)
            self.is_valid(route=route, alpha=alpha)
            self.route = route
        except ValueError:
            raise StopIteration("could not swap point %d and point %d in route %s" % (pos1, pos2, self.route))

    def delay(self, route):
        """
        Returns the delay of the route.
        """
        if len(route) == 2:
            delay = 0
            return delay
        else:
            request_delays = {}
            for point in route:
                if point[1].id not in request_delays:
                    request_delays[point[1].id] = point[0]
                else:
                    request_delays[point[1].id] = abs(request_delays[point[1].id] - point[0])
                    - travel_time(point[1].PU_coordinates, point[1].DO_coordinates)
            assert len(request_delays) == len(route) / 2
            delay = round(sum(request_delays.values()), 3)
            assert delay >= 0
            return delay

    @property
    def individual_stats(self):
        """
        Stats includes per client:
        - delay: the time delay
        - saving: the price saving compared to an individual course
        - earlier_start: the time gap between the requested Pick Up datetime and the real one which is earlier
        """
        on_board = []
        individual_delays = defaultdict(int)
        individual_savings_per = defaultdict(int)
        individual_earlier_starts = defaultdict(int)
        real_times = defaultdict(int)
        coordinates = defaultdict(list)
        for i, point in enumerate(self.route[:-1]):
            time = travel_time(self.route[i][2], self.route[i+1][2])
            cost = haversine(self.route[i][2], self.route[i+1][2])
            if point[1].id not in on_board:
                on_board.append(point[1].id)
                coordinates[point[1].id].append(point[2])
                individual_earlier_starts[point[1].id] = point[1].PU_datetime[1] - point[0]
            else:
                on_board.remove(point[1].id)
                coordinates[point[1].id].append(point[2])

            for r_id in on_board:
                individual_delays[r_id] += time
                individual_savings_per[r_id] += cost / len(on_board)
        coordinates[self.route[-1][1].id].append(self.route[-1][2])

        assert len(individual_delays) == len(individual_savings_per)
        assert len(individual_delays) == len(individual_earlier_starts)
        assert len(on_board) == 1
        assert [len(t) for t in coordinates.values()] == [2 for _ in range(int(len(self.route) / 2))]

        individual_delays_per = copy.deepcopy(individual_delays)
        individual_earlier_starts_per = copy.deepcopy(individual_earlier_starts)
        for r_id in individual_delays.keys():
            time = travel_time(coordinates[r_id][0], coordinates[r_id][1])
            cost = haversine(coordinates[r_id][0], coordinates[r_id][1])

            assert individual_earlier_starts[r_id] >= 0
            individual_earlier_starts_per[r_id] = individual_earlier_starts[r_id]*100 / individual_delays[r_id]

            individual_delays[r_id] -= time
            individual_delays_per[r_id] -= time
            assert individual_delays[r_id] >= 0
            individual_delays_per[r_id] = individual_delays_per[r_id]*100 / time

            individual_savings_per[r_id] -= cost
            assert individual_savings_per[r_id] <= 0
            individual_savings_per[r_id] = abs(individual_savings_per[r_id])*100 / cost

        return individual_delays, individual_delays_per, individual_savings_per, individual_earlier_starts, individual_earlier_starts_per

    def is_valid(self, route, alpha: float) -> bool:
        """
        Checks the validity of the route according to the following constraints :
            1. The capacity constraint
            2. No stops permitted in the taxi's route
            3. The cost constraint
            4. The time windows of the passengers
        """
        served_requests = []
        loading_timeline = [0]
        for i, x in enumerate(route):
            if x[1].id not in served_requests:
                loading_timeline.append(loading_timeline[-1] + 1)
                # 1. The capacity constraint
                if loading_timeline[-1] > self.capacity:
                    return False
                served_requests.append(x[1].id)
            else:
                loading_timeline.append(loading_timeline[-1] - 1)
                # 2. No stops permitted in the taxi's route
                if loading_timeline[-1] == 0 and i < len(route) - 1:
                    return False

        # sanity checks
        assert len(loading_timeline) == len(route) + 1
        assert loading_timeline[-1] == 0

        # 3. The cost constraint
        travel_costs = defaultdict(int)
        coordinates = defaultdict(list)
        on_board = []
        for i, point in enumerate(route[:-1]):
            cost = haversine(route[i][2], route[i+1][2])
            if point[1].id not in on_board:
                on_board.append(point[1].id)
                coordinates[point[1].id].append(point[2])
            else:
                on_board.remove(point[1].id)
                coordinates[point[1].id].append(point[2])
            for r_id in on_board:
                travel_costs[r_id] += cost / len(on_board)
        coordinates[route[-1][1].id].append(route[-1][2])

        assert len(on_board) == 1
        assert [len(t) for t in coordinates.values()] == [2 for _ in range(int(len(route) / 2))]

        for r_id, cost in travel_costs.items():
            if cost > haversine(coordinates[r_id][0], coordinates[r_id][1]) * alpha:
                return False

        # 4. The time windows of the passengers
        served_requests = []
        for i in range(len(route) - 1):
            # updates the delivery datetimes of the route after the insertions
            time_to_next_point = travel_time(route[i][2], route[i+1][2])
            if route[i][0] + time_to_next_point > route[i+1][0]:
                if route[i][1].id not in served_requests: # i source
                    served_requests.append(route[i][1].id)
                    if route[i+1][1].id in served_requests: # i+1 destination
                        route[i+1][0] += time_to_next_point - (route[i+1][0] - route[i][0])
                    else: # i+1 source
                        for j in range(i+1):
                            route[j][0] -= time_to_next_point - route[i+1][0]
                else: # i destination
                    if route[i+1][1].id in served_requests: # i+1 destination
                        route[i+1][0] += time_to_next_point - (route[i+1][0] - route[i][0])
                    else: # i+1 source
                        for j in range(i+1):
                            route[j][0] -= route[i][0] + time_to_next_point            
            else:
                served_requests.append(route[i][1].id)

            # check for violated time window constraints
            sr = []
            for j in range(i+1):
                # source
                if route[j][1].id not in sr and (route[j][0] < route[j][1].PU_datetime[0] or route[j][0] > route[j][1].PU_datetime[1]):
                    return False
                # destination
                if route[j][1].id in sr and route[j][0] > route[j][1].DO_datetime[1]:
                    return False
                sr.append(route[j][1].id)

        # check for the last point of the route
        if route[-1][0] < route[-1][1].DO_datetime[0] or route[-1][0] > route[-1][1].DO_datetime[1]:
            return False
        return True
