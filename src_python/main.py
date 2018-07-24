import argparse
import csv
import logging
import sys
from typing import Callable, Dict, List, Tuple

from request import Request
from solution import Solution
from taxi import Taxi
from utils import request2coordinates, time


def setup():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the dataset of taxi requests.")
    parser.add_argument("-s", "--size", type=int, default=None,
                        help="Maximum size of the dataset to import.")
    parser.add_argument("-t", "--timeframe", type=int, default=1000,
                        help="Time length of a static instance (sec).")
    parser.add_argument("--speed", type=int, default=40,
                        help="Constant speed for the taxis: 40km/h in Santos's paper.")
    parser.add_argument("-c", "--capacity", type=int, default=2,
                        help="Seat capacity of each taxi, driver not included.")
    parser.add_argument("--alpha", type=float, default=1,
                        help="Customer are paying less than an individual ride times alpha.")
    parser.add_argument("--num-GRASP", type=int, default=10,
                        help="Maximum number of iterations of the GRASP heuristic.")
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    return args


def read_dataset(path: str, size: int=None, time_window: int=15) -> List[List[Tuple[float]]]:
    """
    Function to read the input CSV file with the taxi requests.

    param: path: path to the CSV file containg taxi requests, downloaded from
    http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
    For example 2015-January-Yellow.
    There are no more latitude and longitude coordinates after 2016 but only area zones.
    param: size: an upper bound for the number of rows in the CSV we are reading because
        those CSV files count millions of rows.

    return: a list of taxi requests where one row includes Pick-Up and Drop-Off time windows
        plus Pick-Up and Drop-Off coordinates
    [(PU_datetime - 15min, PU_datetime), (DO_datetime, DO_datetime + 15min),
    (PU_longitude, PU_latitude), (DO_longitude, DO_latitude)]
    """
    log = logging.getLogger("reader")
    fin = open(path, newline="")
    try:
        if not size:  # if no size specified, read the all dataset
            size = sum(1 for _ in csv.reader(fin))
        fin.seek(0)
        reader = csv.reader(fin)
        next(reader)  # to not read the headers
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

            PU_datetime = datetime(row[1])
            PU_coordinates = (float(row[5]), float(row[6]))
            DO_coordinates = (float(row[9]), float(row[10]))
            # we do not use the DO datetime from the dataset because we use distances
            # the distance as the crow flies with constant speed
            DO_datetime = PU_datetime + time(PU_coordinates, DO_coordinates)

            request = []
            request.append((PU_datetime - time_window*60, PU_datetime))
            request.append((DO_datetime, DO_datetime + time_window*60))
            request.append(PU_coordinates)
            request.append(DO_coordinates)
            dataset.append(request)

        # sort the requests by Pick-Up datetime
        dataset_sorted = sorted(dataset, key=lambda x: x[0][0])
    finally:
        sys.stderr.write("\n")
        fin.close()
    log.info("Dataset size: %d taxi requests" % len(dataset))
    return dataset_sorted


def initialize_requests(dataset, timeframe: int) -> List:
    log = logging.getLogger("initialize_requests")
    requests = []
    t_0 = dataset[0][0][0]
    log.info("Origin datetime : %d" % t_0)
    for i, req in enumerate(dataset):
        request = Request(req, i+1)
        if request.PU_datetime()[0] < t_0 + timeframe:
            requests.append(request)
        else:
            break
    log.info("Timeframe : %d sec" % timeframe)
    log.info("Number of taxi requests : %d" % len(requests))
    return requests


def path_relinking(s1, s2):
    """
    TODO
    """


def main():
    args = setup()
    dataset = read_dataset(args.input, args.size)
    requests = initialize_requests(dataset, timeframe=args.timeframe)

    print("Starting GRASP iterations...")
    for i in range(args.num_GRASP):
        print("----- Iteration :", i+1)
        solution = Solution(requests)
        solution.build_initial_solution(beta=0.5)
        for i, taxi in enumerate(solution.taxis):
            print("number of requests served by taxi %d : %d" % (i, int(len(taxi.route)/2)))
        nb_shared_rides = solution.compute_objective_function()
        print("Number of shared rides :", nb_shared_rides)
        print("Local search performed...")
        next_solution = solution.local_search()
        #next_solution = Neighborhood(solution)
        #print("2 :", next_solution.taxis)
        #next_solution.local_search()
        #next_nb_shared_rides = next_solution.compute_objective_function(nb_requests)
        #print("Number of shared rides :", next_nb_shared_rides)
        # solution = path_relinking(s1, s2)
        # solution.local_search()


if __name__ == "__main__":
    sys.exit(main())
