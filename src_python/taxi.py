from typing import Callable, List


class Taxi:

    def __init__(self, request, capacity: int=2, speed: int=40):
        """
        list of request indices in the same order as the taxi drives them.
        positive indices stand for PU points when negative indices stand for DO points.
        """
        B_source = request.PU_datetime()[0]
        B_destination = request.DO_datetime()[0]
        # e.g. [(B3_source, request_3), (B6_source, request_6), (B3_destination, request_3), (B6_destination, request_6)]
        self.route = [(B_source, request), (B_destination, request)]
        self.capacity = capacity
        self.speed = speed

    def insert(self, request: Callable):
        """
        TODO
        """
        for pu in range(len(self.route) + 1):
            self.route.insert(pu, (request.PU_datetime()[0], request))
            for do in range(pu + 1, len(self.route) + 1):
                try:
                    self.route.insert(do, (request.DO_datetime()[0], request))
                    self.is_valid()
                    self.update_delivery_datetimes()
                    return
                except Exception:
                    continue
        raise Exception("could not insert request %d into the route %s" % (request.id, self.route))

    def remove(self, request: Callable):
        """
        TODO
        """
        self.route = [req for req in self.route if req[1] == request]

    def get_loading_timeline(self) -> List[int]:
        served_request = []
        loading_timeline = [0]
        for x in self.route:
            if x[1] not in served_request:
                loading_timeline.append(loading_timeline[-1] + 1)
                served_request.append(x[1])
            else:
                loading_timeline.append(loading_timeline[-1] - 1)

        # for future tests
        assert len(loading_timeline) == len(self.route) + 1
        assert loading_timeline[-1] == 0

        return loading_timeline

    def is_valid(self) -> bool:
        """
        check:
        1. the capacity constraint
        2. the time windows of the passengers
        """

        # check the capacity constraint
        loading_timeline = self.get_loading_timeline()
        if max(loading_timeline) > self.capacity:
            raise Exception("The maximum number of passengers served at the same time %d"
                            "exceed the taxi capacity %d" % (max(loading_timeline, self.capacity)))

        # check the time windows of the passengers
        served_requests = []
        for i in range(len(self.route)):

            # source point
            if self.route[i][1] not in served_requests and self.route[i][0] > self.route[i][1].PU_datetime()[1]:
                raise Exception("The taxi can not pick up the client %d after the end of the time window %d"
                                % (self.route[i][1].id, self.route[i][1].PU_datetime()[1]))
            
            # destination point
            elif self.route[i][1] in served_requests and self.route[i][0] > self.route[i][1].DO_datetime()[1]:
                raise Exception("The taxi can not drop off the client %d after the end of the time window %d"
                                % (self.route[i][1].id, self.route[i][1].DO_datetime()[1]))
            served_requests.append(self.route[i][1])

    def update_delivery_datetimes(self):
        served_requests = []
        for i in range(len(self.route)):
            # source point
            if self.route[i][1] not in served_requests:
                if self.route[i][0] < self.route[i][1].PU_datetime()[0]:
                    self.route[i][0] = self.route[i][1].PU_datetime()[0]
            # destination point
            else:
                if self.route[i][0] < self.route[i][1].DO_datetime()[0]:
                    self.route[i][0] = self.route[i][1].DO_datetime()[0]
