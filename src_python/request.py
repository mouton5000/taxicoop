from typing import Tuple


class Request:
    """
    Class built to access characteristics of the requests in the dataset.
    param: row: one request extracted from the dataset :
    [(PU_datetime - 15min, PU_datetime), (DO_datetime, DO_datetime + 15min),
    (PU_longitude, PU_latitude), (DO_longitude, DO_latitude)]
    param: passengers: number of passengers involved in the request.

    """
    def __init__(self, row, id: int, passengers: int=1):
        self._row = row
        self.id = id
        self.passengers = passengers

    def PU_datetime(self) -> Tuple[int]:
        return self._row[0]

    def DO_datetime(self) -> Tuple[int]:
        return self._row[1]

    def PU_coordinates(self) -> Tuple[float]:
        return self._row[2]

    def DO_coordinates(self) -> Tuple[float]:
        return self._row[3]
