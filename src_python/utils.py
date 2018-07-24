from typing import Callable, Tuple

from haversine import haversine


def time(u: Tuple[float], v: Tuple[float], distance: Callable=haversine,
         speed: int=40/3600) -> float:
    if isinstance(u, tuple) and isinstance(v, tuple):
        return distance(u, v) / speed
    else:
        raise TypeError("u and v must be integers, got %s and %s" % (type(u), type(v)))


def request2coordinates(request, served_requests):
    if request not in served_requests:
        v_point = request.PU_coordinates()
        served_requests.append(request)
    else:
        v_point = request.PU_coordinates()
    return v_point, served_requests
