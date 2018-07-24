import random
from typing import Callable, Dict, List

from taxi import Taxi
from utils import request2coordinates, time


class Solution:
    def __init__(self, requests: List):
        self.requests = requests
        self.nb_requests = len(requests)
        self.taxis = []

    def build_initial_solution(self, beta: float):
        """
        TODO
        """
        self.taxis.append(Taxi(self.requests.pop(0)))
        while len(self.requests) > 0:
            # the lower mu is, the better
            mu = self.get_greedy_function(self.taxis[-1], self.requests)
            mu_max = mu[max(mu, key=mu.get)]
            mu_min = mu[min(mu, key=mu.get)]
            RCL_UB = mu_min + beta * (mu_max - mu_min)
            RCL = []
            for request, mu_req in mu.items():
                if mu_req <= RCL_UB:
                    RCL.append(request)
            while len(RCL) > 0:
                request = RCL.pop()
                try:
                    self.taxis[-1].insert(request)
                    self.requests.remove(request)
                    break
                except Exception:
                    continue

            # no requests can be inserted into the route of the last taxi, so we take a new taxi
            if len(RCL) == 0:
                try:
                    self.taxis.append(Taxi(self.requests.pop(0)))
                except IndexError:
                    continue

    def get_greedy_function(self, taxi: Callable, requests: List) -> Dict:
        mu = {}
        for u in requests:  # u is the request to insert in the taxi's route
            delta_source = {}
            delta_destination = {}
            served_requests = []
            for i in range(len(taxi.route)-1):  # we do not care if v is a source point or not
                v = taxi.route[i]
                v_next = taxi.route[i+1]

                v_point, served_requests = request2coordinates(v[1], served_requests)
                v_next_point, served_requests = request2coordinates(v_next[1], served_requests)

                # u source
                # >=
                if v[0] + time(v_point, u.PU_coordinates()) >= u.PU_datetime()[0]:
                    delta_source[u.id] = time(v_point, u.PU_coordinates()) \
                        + time(u.PU_coordinates(), v_next_point) - time(v_point, v_next_point)
                # <
                else:
                    delta_source[u.id] = u.PU_datetime()[0] - v[0] \
                        + time(u.PU_coordinates(), v_next_point) - time(v_point, v_next_point)

                # u destination
                # <=
                if v[0] + time(v_point, u.DO_coordinates()) <= u.DO_datetime()[1]:
                    delta_destination[u.id] = time(v_point, u.DO_coordinates()) \
                        + time(u.DO_coordinates(), v_next_point) - time(v_point, v_next_point)
                # >
                else:
                    delta_destination[u.id] = 10000

            # mu[u] is the sum of the 2 minimal delays to insert u in the taxi's route
            mu[u] = min(delta_source, key=lambda x: delta_source.get(x)) \
                + min(delta_destination, key=lambda x: delta_destination.get(x))
        return mu

    def compute_objective_function(self) -> int:  # number of shared rides to maximize
        private_rides = [taxi for taxi in self.taxis if len(taxi.route)/2 == 1]
        return self.nb_requests - len(private_rides)

    def local_search(self):
        """
        TODO
        """
        next_solution = self.switch_requests()
        return next_solution
        
    def switch_requests(self, nb_attempts: int=1):
        """
        Operation used in Santos's both 2013 and 2015 paper.
        Permutation of two requests from different routes.
        """
        for _ in range(nb_attempts):
            try:
                taxi1, taxi2 = random.sample(self.taxis, 2)

                request1 = random.sample(taxi1.route, 1)
                taxi1.remove(request1[0][1])
                request2 = random.sample(taxi2.route, 1)
                taxi2.remove(request2[0][1])
                
                taxi1.insert(request2[0][1])
                taxi2.insert(request1[0][1])
                return self
            except Exception:
                continue
        raise Exception("%d attempts to switch 2 requests reached"
                        " without finding any valid routes." % nb_attempts)

    def switch_points(self):
        """
        Operation used only in Santos's 2013 paper.
        Permutation of two consecutive points of the same route.
        """

    def remove_insert_request(self):
        """
        Operation used only in Santos's 2013 paper.
        Removal of some request from some route and attempt to insert,
        in this route, some request that was not served.
        """
