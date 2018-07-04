import argparse
import csv
import logging
import os
import sys
from clint.textui import progress
from typing import Tuple, List

from haversine import haversine
import numpy


def setup():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the dataset of taxi requests.")
    parser.add_argument("-s", "--size", type=int, default=None,
                        help="Maximum size of the dataset to import.")
    parser.add_argument("-v", "--speed", type=int, default=40,
                        help="Constant speed for the taxis: 40km/h in Santos's paper.")
    parser.add_argument("-c", "--capacity", type=int, default=2,
                        help="Seat capacity of each taxi, driver not included.")
    parser.add_argument("--alpha", type=float, default=1,
                        help="Customer are paying less than an individual ride times alpha.")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Maximum number of iterations in the algorithm.")
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    return args


def read_dataset(path: str, size: int=None, time_window: int=15) -> List:
    """
    Function to read the input CSV file with the taxi requests.

    param: path to the CSV file containg taxi requests, downloaded from
    http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
    For example 2015-January-Yellow.
    There are no more latitude and longitude coordinates after 2016 but only area zones.
    param: size is an upper bound for the number of rows in the CSV we are reading because
        those CSV files count millions of rows.

    return: a list of taxi requests where one row includes Pick-Up and Drop-Off time windows
        plus Pick-Up and Drop-Off coordinates
    [(PU_datetime - 15min, PU_datetime), (DO_datetime, DO_datetime + 15min),
    (PU_longitude, PU_latitude), (DO_longitude, DO_latitude)]
    """
    log = logging.getLogger("reader")
    fin = open(path, newline="")
    try:
        if not size: # if no size specified, read the all dataset
            size = sum(1 for _ in csv.reader(fin))
        fin.seek(0)
        reader = csv.reader(fin)
        next(reader) # to not read the headers
        dataset = []

        def datetime(val: str) -> int:
            datetime = val.split()
            timestamp = datetime[1].split(":")
            timestamp = [int(x) for x in timestamp]
            hour, minutes, seconds = tuple(timestamp)
            time_absolute = hour*3600 + minutes*60 + seconds
            return time_absolute

        for i, row in enumerate(reader):
            if i % 1000 == 0:
                sys.stderr.write("%d\r" % i)
            if i >= size:
                break
            request = []
            request.append((datetime(row[1]) - time_window*60, datetime(row[1])))
            request.append((datetime(row[2]), datetime(row[2]) + time_window*60))
            request.append((float(row[5]), float(row[6])))
            request.append((float(row[9]), float(row[10])))
            # some requests are corrupted, we don't add them to the dataset
            if request[0][1] < request[1][0]:
                dataset.append(request)
    finally:
        sys.stderr.write("\n")
        fin.close()
    return dataset


class Request:
    """
    Class built to access characteristics of the requests in the dataset.
    param: row: one request extracted from the dataset :
    [(PU_datetime - 15min, PU_datetime), (DO_datetime, DO_datetime + 15min),
    (PU_longitude, PU_latitude), (DO_longitude, DO_latitude)]
    param: passengers: number of passengers involved in the request.

    """
    def __init__(self, row, passengers=1):
        self._row = row
        self.passengers = passengers

    def PU_datetime(self):
        return self._row[0]

    def DO_datetime(self):
        return self._row[1]

    def PU_coordinates(self):
        return self._row[2]

    def DO_coordinates(self):
        return self._row[3]


def build_distance_matrix(dataset, distance=haversine):
    """
    Builds the adjacency matrix of the graph.
    As it is specified in Santos's paper from 2015, the shape of the matrix is 2*N
    where N is the number of requests considered in the static resolution.
    The weights of the edges are the distance between 2 pairs of coordinates.
    Like in Santos's paper, we choose the Haversine distance and used the following
    Python package for its calculation https://pypi.org/project/haversine/
    """
    nb_requests = len(dataset)
    A = numpy.zeros((2*nb_requests, 2*nb_requests))
    for i, row_i in progress.bar(enumerate(dataset), expected_size=nb_requests):
        Request_i = Request(row_i)
        for j, row_j in enumerate(dataset):
            Request_j = Request(row_j)
            # Fill the 4 blocks of the adjacency matrix
            if i != j:  # The main diagonal and the one at the bottom-left part of the matrix are null
                # Avoid distance calculations based on time windows constraints
                if Request_i.PU_datetime()[0] >= Request_j.PU_datetime()[1]:
                    A[i][j] = distance(Request_i.PU_coordinates(), Request_j.PU_coordinates())
                if Request_i.DO_datetime()[0] >= Request_j.DO_datetime()[1]:
                    A[i + nb_requests][j + nb_requests] = distance(Request_i.DO_coordinates(), Request_j.DO_coordinates())
                if Request_i.DO_datetime()[0] >= Request_j.PU_datetime()[1]:
                    A[i + nb_requests][j] = distance(Request_i.DO_coordinates(), Request_i.PU_coordinates()) 
            if Request_i.PU_datetime()[0] >= Request_j.DO_datetime()[1]:
                A[i][j + nb_requests] = distance(Request_i.PU_coordinates(), Request_j.DO_coordinates())
    return A


def main():
    args = setup()
    dataset = read_dataset(args.input, args.size)
    print("len_dataset :", len(dataset))
    A = build_distance_matrix(dataset)


if __name__ == "__main__":
    sys.exit(main())
