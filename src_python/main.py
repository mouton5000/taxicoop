import argparse
import copy
import csv
import logging
import pickle
import sys
import time
from typing import Callable, Dict, List, Tuple

from matplotlib import pyplot as plt

from request import Request
from solution import Solution
from taxi import Taxi
from utils import request2coordinates, travel_time


def setup():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str,
                        help="Path to the dataset of taxi requests.")
    parser.add_argument("--checkpoint-dataset", type=str, required=True,
                        help="Path to the dataset of taxi requests.")
    parser.add_argument("-s", "--size", type=int, default=None,
                        help="Maximum size of the dataset to import.")
    parser.add_argument("-w", "--time-window", type=int, default=15,
                        help="Length of the time windows for both pick up and drop off points.")
    parser.add_argument("--data-saved", action="store_true",
                        help="Load a static instance of the DARP-M with 4298 requests on 30 sec.")
    parser.add_argument("-t", "--timeframe", type=int, default=1000,
                        help="Time length of a static instance (sec).")
    parser.add_argument("--speed", type=int, default=40,
                        help="Constant speed for the taxis: 40km/h in Santos's paper.")
    parser.add_argument("-c", "--capacity", type=int, default=2,
                        help="Seat capacity of each taxi, driver not included.")
    parser.add_argument("--alpha", type=float, default=1,
                        help="Customer are paying less than an individual ride times alpha.")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Size of the Restricted Candidate List (RCL) in the insertion method.")
    parser.add_argument("--num-GRASP", type=int, default=10,
                        help="Maximum number of iterations of the GRASP heuristic.")
    parser.add_argument("--num-local-search", type=int, default=10,
                        help="Maximum number of iterations in the local search.")
    parser.add_argument("--insertion-method", type=str, default="IA", choices=("IA", "IB"),
                        help="Type of insertion method to use."
                              "IA stands for the exhaustive method."
                              "IB stands for the heuristic method.")
    parser.add_argument("--test-size", type=int, default=200,
                        help="Number of requests to consider to test the algorithm.")
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
            DO_datetime = PU_datetime + travel_time(PU_coordinates, DO_coordinates)

            request = []
            request.append((PU_datetime - time_window*60, PU_datetime))
            request.append((DO_datetime, DO_datetime + time_window*60))
            request.append(PU_coordinates)
            request.append(DO_coordinates)
            
            # filtering
            if PU_coordinates[0] != 0 and DO_coordinates[0] != 0 and DO_datetime - PU_datetime < 12*3600:
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
        if request.PU_datetime[0] < t_0 + timeframe:
            requests.append(request)
        else:
            break
    log.info("Timeframe : %d sec" % timeframe)
    log.info("Number of taxi requests : %d" % len(requests))
    return requests


def main():
    args = setup()
    if args.data_saved:
        with open(args.checkpoint_dataset, "rb") as f:
            requests = pickle.load(f)
    else:
        dataset = read_dataset(args.input, args.size, args.time_window)
        requests = initialize_requests(dataset, timeframe=args.timeframe)
        with open(args.checkpoint_dataset, "wb") as f:
            pickle.dump(requests, f)

    print("Starting GRASP iterations...")
    time_start = time.clock()
    elite_solution = None
    for i in range(args.num_GRASP):
        print()
        print("----- Iteration %d -----" % (i+1))
        init_solution = Solution(requests[:args.test_size])
        init_solution.build_initial_solution(args.insertion_method, beta=args.beta)

        init_obj = init_solution.compute_obj
        print("Obj init :", init_obj)
        if init_obj == init_solution.nb_requests:
            elite_solution = copy.deepcopy(init_solution)
            elite_obj = init_obj
            break

        print("1. Local Search :", init_obj)
        ls_solution, ls_obj = init_solution.local_search(args.insertion_method, args.num_local_search)
        if ls_obj == ls_solution.nb_requests:
            elite_solution = copy.deepcopy(ls_solution)
            elite_obj = ls_obj
            break
        
        if elite_solution:
            print("2. Path Relinking :", ls_obj)
            pr_solution, pr_obj = ls_solution.path_relinking(elite_solution, args.insertion_method)
            print("3. Second Local Search :", pr_obj)
            ls_pr_solution, ls_pr_obj = pr_solution.local_search(args.insertion_method, args.num_local_search)
            if ls_pr_obj > elite_obj:
                elite_solution = copy.deepcopy(ls_pr_solution)
                elite_obj = ls_pr_obj
        else:
            elite_solution = copy.deepcopy(ls_solution)
            elite_obj = ls_obj
        print()
        print("Elite obj :", elite_obj)

    print("Best obj after %d iterations of the GRASP heuristic : %d" % (args.num_GRASP, elite_obj))

    time_elapsed = (time.clock() - time_start)
    print("Computation time :", round(time_elapsed, 2))


if __name__ == "__main__":
    sys.exit(main())
