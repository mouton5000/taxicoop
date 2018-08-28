from collections import defaultdict
import copy
import random
import sys
from typing import Callable, Dict, List, Set, Tuple, Union

from taxi import Taxi
from utils import request2coordinates, travel_time


class Solution:
    def __init__(self, requests: List):
        self.requests = {r.id:r for r in requests}
        self.nb_requests = len(requests)
        self.taxis = []

    def build_initial_solution(self, insertion_method, beta: float, limit_RCL=0.2):
        """
        Builds the initial greedy solution at the beginning of each GRASP iteration.
        """
        requests = copy.deepcopy(list(self.requests.keys()))
        random.shuffle(requests)
        random_id = requests.pop(0)
        self.taxis.append(Taxi(self.requests[random_id]))
        while len(requests) > 0:
            # sys.stderr.write("requests pending : %d\r" % len(self.requests))
            # the lower mu is, the better
            mu = self.get_greedy_function(self.taxis[-1], requests)
            mu_max = mu[max(mu, key=mu.get)]
            mu_min = mu[min(mu, key=mu.get)]
            RCL_UB = mu_min + beta * (mu_max - mu_min)
            RCL = []
            for r_id, mu_r in mu.items():
                if mu_r <= RCL_UB:
                    RCL.append(r_id)
            it = 0
            max_it = round(len(RCL) * limit_RCL)
            while len(RCL) > 0:
                if it > max_it:
                    break
                r_id = RCL.pop(0)
                it += 1
                try:
                    self.taxis[-1].insert(self.requests[r_id], method=insertion_method)
                    requests.remove(r_id)
                    break
                except Exception:
                    continue

            # no requests can be inserted into the route of the last taxi, so we take a new taxi
            if len(RCL) == 0 or it > max_it:
                try:
                    random_id = requests.pop(0)
                    self.taxis.append(Taxi(self.requests[random_id]))
                except IndexError:
                    continue

    def get_greedy_function(self, taxi: Callable, requests: List) -> Dict:
        """
        First greedy function detailed in Santos 2015 based on the minimum delay.
        """
        mu = {}
        for r_id in requests:  # u is the request to insert in the taxi's route
            r = self.requests[r_id]
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
                if v[0] + travel_time(v_point, r.PU_coordinates) >= r.PU_datetime[0]:
                    delta_source[r_id] = travel_time(v_point, r.PU_coordinates) \
                        + travel_time(r.PU_coordinates, v_next_point) - travel_time(v_point, v_next_point)
                # <
                else:
                    delta_source[r_id] = r.PU_datetime[0] - v[0] \
                        + travel_time(r.PU_coordinates, v_next_point) - travel_time(v_point, v_next_point)

                # u destination
                # <=
                if v[0] + travel_time(v_point, r.DO_coordinates) <= r.DO_datetime[1]:
                    delta_destination[r_id] = travel_time(v_point, r.DO_coordinates) \
                        + travel_time(r.DO_coordinates, v_next_point) - travel_time(v_point, v_next_point)
                # >
                else:
                    delta_destination[r_id] = 10000

            # mu[u] is the sum of the 2 minimal delays to insert u in the taxi's route
            mu[r_id] = min(delta_source, key=lambda x: delta_source.get(x)) \
                + min(delta_destination, key=lambda x: delta_destination.get(x))
        return mu

    @property
    def compute_obj(self) -> int:
        """
        Computes the objective function to maximize i.e. the number of shared rides.
        """
        private_rides = [taxi for taxi in self.taxis if len(taxi.route)/2 == 1]
        return self.nb_requests - len(private_rides)

    @property
    def pending_requests(self):
        """
        Returns the requests that are not assigned i.e. assigned to a individual ride in the solution.
        """
        pending_requests = set()
        for taxi in self.taxis:
            if len(taxi.route) == 2:
                pending_requests.add(taxi.route[0][1].id)
        return pending_requests

    def local_search(self, insertion_method, max_iter: int=10, nb_swap: int=3):
        """
        Algorithm that performs the local search of the GRASP heuristic.
        At each iteration :
            1. We try to insert all pending requests in random routes of the solution.
            2. If it improves the objective function, we pass to the next iteration.
            3. If not, we perform a nb_swap number of random swap operations in the solution.
        At any time the upper bound is reached, we get out of the local search.
        """
        current_solution = copy.deepcopy(self)
        current_obj = current_solution.compute_obj
        for i in range(max_iter):
            print("  -- it ", i)
            next_solution = current_solution.insert_requests(current_solution.pending_requests, insertion_method)
            next_obj = next_solution.compute_obj
            if next_obj > current_obj:
                current_solution = copy.deepcopy(next_solution)
                current_obj = next_obj
                print("     obj :", current_obj)
                if current_obj == self.nb_requests: # upper bound
                    break
            else:
                for s in range(nb_swap):
                    #print("     swap it :", s)
                    current_solution = current_solution.swap_requests(insertion_method)
        return current_solution, current_obj

    def path_relinking(self, initial_solution, insertion_method):
        """
        Algorithm that performs the path relinking from s1 to s2 of the GRASP heuristic, cf Santos 2015.
            0. We find the requests that could be inserted i.e. the commun pending requests in s1 and s2.
               If this set of request is null, there is no need to perform any path relinking
            1. We merge the pending requests of s1 that are associated in s2.
            2. We try to insert the pending requests in s1 in the routes they are likely to be accepted according to s2
               i.e. in the routes that contain requests they are associated with in s2.
            3. We perform a series of swap of the adjacent requests of those previous requests in order to reduce 
               the delay of target routes wher to insert the pending requests.
            4. We apply step 2 again.
        """
        initial_pending_requests = initial_solution.pending_requests
        self_pending_requests = self.pending_requests
        initial_pending_requests -= self_pending_requests

        # no need to do the path relinking if the pending requests in s1 are also pending in s2
        if len(initial_pending_requests) == 0:
            return initial_solution, initial_solution.compute_obj

        self_routes, other_requests = self.get_taxis_from_requests(initial_pending_requests)
        requests_to_merge = set.intersection(set(other_requests.keys()),
                                             set(i for s in other_requests.values() for i in s))        
        if requests_to_merge:
            for req_id in requests_to_merge:
                initial_pending_requests.remove(req_id)
            initial_solution = initial_solution.merge_pending_requests(requests_to_merge, other_requests, insertion_method)
        print("     1. obj :", initial_solution.compute_obj)

        initial_solution, initial_pending_requests = initial_solution.insert_requests(initial_pending_requests, insertion_method, other_requests)
        print("     2. obj :", initial_solution.compute_obj)
        initial_solution = initial_solution.guided_swap(other_requests.values())
        initial_solution, initial_pending_requests = initial_solution.insert_requests(initial_pending_requests, insertion_method, other_requests)
        print("     3. obj :", initial_solution.compute_obj)
        
        return initial_solution, initial_solution.compute_obj

    def merge_pending_requests(self, requests_to_merge, other_requests, insertion_method):
        """
        Function used in the path relinking that merges the pending requests of the s1 solution
        that are associated in the s2 solution.
        """
        nb_requests_to_merge = len(requests_to_merge)
        routes_dict, _ = self.get_taxis_from_requests(requests_to_merge)
        obj_improvments = 0
        while len(requests_to_merge) > 0:
            r1 = requests_to_merge.pop()
            obj_improvments += 1
            for r2 in other_requests[r1]:
                if r2 in requests_to_merge:
                    routes_dict[r1].insert(self.requests[r2], method=insertion_method)
                    self.remove_pending_request(r2)
                    requests_to_merge.remove(r2)
                    obj_improvments += 1
        assert obj_improvments == nb_requests_to_merge
        return self

    def get_potential_routes(self, other_requests: Dict):
        """
        Returns a dict where the key is a request (the same as the key in other_requests) 
        and its value is the list of routes that are serving the corresponding requests in other_requests.
        """
        potential_routes = {}
        for r_id, other_reqs in other_requests.items():
            routes_dict, _ = self.get_taxis_from_requests(other_reqs)
            potential_routes[r_id] = list(routes_dict.values())
        return potential_routes

    def insert_requests(self, requests: List, insertion_method, other_requests=None, nb_attempts=1):
        """
        Inserts the requests provided as input in :
            1. the routes that already serve the requests in other_requests if provided.
            2. random routes of the solution if no other_requests is provided.
        """
        if other_requests is not None:
            requests_inserted = []
            potential_routes = self.get_potential_routes(other_requests)
            for req_id in requests:
                for taxi in potential_routes[req_id]:
                    try:
                        taxi.insert(self.requests[req_id], method=insertion_method)
                        self.remove_pending_request(req_id)
                        improvment = True
                        requests_inserted.append(req_id)
                    except Exception:
                        continue
            for req_id in requests_inserted:
                requests.remove(req_id)
            return self, requests

        else:
            for req_id in requests:
                request = self.requests[req_id]
                for j in range(nb_attempts):
                    taxi = random.sample(self.taxis, 1)[0]
                    if req_id != taxi.route[0][1].id:
                        try:
                            taxi.insert(request, method=insertion_method)
                            self.remove_pending_request(req_id)
                            break
                        except Exception:
                            continue
            return self

    def remove_pending_request(self, req_id):
        """
        Removes pending requests (requests served in an individual route) from a solution,
        usually because those requests have just been assigned to existing routes of the solution.
        """
        nb_taxis = len(self.taxis)
        self.taxis = [taxi for taxi in self.taxis if not (len(taxi.route) == 2 and taxi.route[0][1].id == req_id)]
        assert len(self.taxis) == nb_taxis - 1

    def guided_swap(self, requests: List):
        """
        Function used in the path relinking to swap each element of requests with other random requests
        according to the one direction swap, cf 3. in swap_requests() 
        """
        adjacent_requests = set([r for sublist in requests for r in sublist])
        init_routes, init_other_requests = self.get_taxis_from_requests(adjacent_requests)
        nb_succes = 0
        for sublist in init_other_requests.values():
            for r in sublist:
                try:
                    self = self.swap_requests(insertion_method, requests=[r])
                    nb_succes += 1
                    break
                except Exception:
                    continue
        print("      -- nb of guided swap :", nb_succes)
        return self
        
    def swap_requests(self, insertion_method, nb_attempts: int=1, requests=None, delay_improvment=0):
        """
        Operation used in both Santos 2013 and 2015 : permutation of two requests from different routes.
        1. If no input requests are provided as input, we try nb_attemps time to swap 2 random requests from the 10 most constrained requests.
        2. If 2 requests are provided, we try to swap those 2.
        3. If only one request is provided, we try nb_attempts time to swap this request with another random request,
            and we keep the swap only if we reduce the dely of the first request's route by delay_improvment.
        """
        if requests is None:
            for i in range(nb_attempts):
                delays = copy.deepcopy(self.delays)
                taxi_id1, taxi_id2 = random.sample(delays[:10], 2)
                try:
                    solution = copy.deepcopy(self)
                    taxi1 = solution.taxis[taxi_id1[0]]
                    taxi2 = solution.taxis[taxi_id2[0]]
                    request1 = random.sample(taxi1.route, 1)
                    request2 = random.sample(taxi2.route, 1)
                    taxi1.remove(request1[0][1])
                    taxi2.remove(request2[0][1])
                    taxi1.insert(request2[0][1], method=insertion_method)
                    taxi2.insert(request1[0][1], method=insertion_method)
                    #print("SWAP %d and %d" % (request1[0][1].id, request2[0][1].id))
                    self = solution
                    return self
                except Exception:
                    continue
            #print("%d attempts to swap 2 requests without success" % nb_attempts)
            return self

        elif len(requests) == 2:
            try:
                solution = copy.deepcopy(self)
                taxis = solution.get_taxis_from_requests(requests)
                taxis[requests[0]].remove(requests[0])
                taxis[requests[1]].remove(requests[1])
                taxis[requests[0]].insert(self.requests[requests[1]], method=insertion_method)
                taxis[requests[1]].insert(self.requests[requests[0]], method=insertion_method)
                self = solution
                return self
            except Exception:
                print("Failed to swap request %d and request %d" % (requests[0], requests[1]))
                pass

        elif len(requests) == 1:
            for i in range(nb_attempts):
                try:
                    solution = copy.deepcopy(self)
                    taxi1, _ = solution.get_taxis_from_requests(requests)
                    taxi1 = taxi1[requests[0]]
                    previous_delay = taxi1.delay(taxi1.route)
                    taxi2 = random.sample(solution.taxis, 1)[0]
                    if taxi2 == taxi1 or len(taxi2.route) == 2:
                        raise Exception("No swap between 2 same routes or with an individual route")
                    request2 = random.sample(taxi2.route, 1)[0][1]
                    taxi1.remove(requests[0])
                    taxi2.remove(request2.id)
                    taxi1.insert(request2, method=insertion_method)
                    taxi2.insert(self.requests[requests[0]], method=insertion_method)
                    if taxi1.delay(taxi1.route) < previous_delay - delay_improvment:
                        self = solution
                        #print("SWAP MAIN %d avec %d" % (requests[0], request2.id))
                        return self
                except Exception:
                    continue
            raise Exception("%d attempts to swap 1 request with a random one without success" % nb_attempts)

        else:
            print("Wrong input :", requests)


    def get_taxis_from_requests(self, requests: Union[List, Set]) -> Tuple[Dict]:
        """
        Function used in the path relinking that returns :
            1. The routes of the current solution serving the requests provided as input.
            2. The other requests served in the same route as the requests provided as input.
        """
        taxis_serving_requests = {}
        other_requests = defaultdict(set)
        for taxi in self.taxis:
            for r_id in requests:
                if any(request[1].id == r_id for request in taxi.route):
                    taxis_serving_requests[r_id] = taxi
                    for point in taxi.route:
                        if point[1].id != r_id:
                            other_requests[r_id].add(point[1].id)

        assert len(taxis_serving_requests) == len(requests)
        return taxis_serving_requests, other_requests        
    
    @property
    def delays(self):
        """
        Returns the delays of the taxi's route, sorted by decreasing order.
        Thus, the first element of the list corresponds to the most constrained route
        i.e. a route likely to recieve a useful swap.
        """
        delays = []
        for i, taxi in enumerate(self.taxis):
            delays.append((i, taxi.delay(taxi.route)))
        delays.sort(key=lambda delay: delay[1], reverse=True)
        return delays

    def swap_points(self, nb_attempts: int=1):
        """
        Operation used only in Santos's 2013 paper.
        Permutation of two consecutive points of the same route.
        """
        for _ in range(nb_attempts):
            try:
                taxi = random.sample(self.taxis, 1)
                pos1, pos2 = random.sample(range(len(taxi)), 2)
                taxi.swap(pos1, pos2)
                return self
            except Exception:
                continue
        raise Exception("%d attempts to swap 2 points of a same request"
                        " without finding any valid route." % nb_attempts)

    def check_no_requests_served_twice(self):
        """
        Test function that returns an exception if at leat one request is served twice in the solution.
        """
        served_requests = [p[1].id for taxi in self.taxis for p in taxi.route]
        if not len(served_requests) / 2 == len(set(served_requests)):
            raise Exception("%d requests are served twice" % (len(served_requests) / 2 - len(set(served_requests))))
