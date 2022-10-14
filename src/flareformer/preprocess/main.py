"""Preprocess physical features and magnetogram images for model input"""
import argparse
from argparse import Namespace
from time import sleep

import numpy as np
import ray
from preprocess.datasets import detect_year_sections, get_image, read_jsonl, split_dataset
from preprocess.utils import identity, make_prefix, sub_1h
from tqdm import tqdm


@ray.remote
def make_yearwise_dataset(
    year: str, data: list, data_type: str, args: Namespace, preprocess=identity, need_save: bool = False
):
    """
    Make dataset for each year
    """
    dataset = []
    print(f"> year: {year}")
    for sample in (pbar := tqdm(data)):
        pbar.set_description(f"> Processing {year}")
        value = sample[data_type]
        dataset.append(preprocess(value))

    if need_save:
        prefix = make_prefix(args.output_path, year, "magnetogram")
        suffix = f"_{args.size}" if args.size != 512 else ""
        path = f"{prefix}{suffix}.npy"
        print(f"> Saved {path}")
        np.save(path, dataset)
        dataset.clear()  # avoid memory leak

    return year, dataset


def make_dataset(data_type: str, database, args: Namespace, preprocess=identity, need_save: bool = False):
    """
    Make dataset (parallel processing)
    """

    print("Execute ray.remote ... ")
    sleep(2)
    process = [
        make_yearwise_dataset.remote(year, data, data_type, args, preprocess, need_save=need_save)
        for year, data in database.items()
    ]
    results = ray.get(process)
    datasets = {item[0]: item[1] for item in results}
    return datasets


def get_window(num, times, current_time, horizon):
    """
    Get window index of time
    """
    s = max(num - horizon + 1, 0)
    index = list(range(s, num)) + [num]
    candidate_time = times[max(s, 0) : num]
    candidate_time.append(current_time)

    ids = []
    time = current_time
    for i in range(horizon):
        if time in candidate_time:
            ids.append(index[candidate_time.index(time)])
        else:
            ids.append(ids[i - 1])
        time = sub_1h(time)

    return ids


def main():
    """
    Execute main process.
    """
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="data/ft_database_all17.jsonl")
    parser.add_argument("--output_path", default="data")
    parser.add_argument("--start_year", default=2010)
    parser.add_argument("--end_year", default=2017)
    parser.add_argument("--horizon", default=48)
    parser.add_argument("--size", default=256)
    parser.add_argument("--magnetogram", action="store_true")
    parser.add_argument("--aia131", action="store_true")
    parser.add_argument("--aia1600", action="store_true")
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--window", action="store_true")

    args = parser.parse_args()
    database_path = args.database_path
    output_path = args.output_path

    # read database
    print(f"Reading database from {database_path}")
    jsonl = read_jsonl(database_path)
    sections, years = detect_year_sections(jsonl)
    database = split_dataset(jsonl, sections)  # dict

    print("Initialize ray for modern parallel and distributed python ... ")
    ray.init(num_cpus=4)  # avoid memory leak

    # Magnetogram Image
    if args.magnetogram:
        print("Prepare magnetogram images ... ")
        _ = make_dataset("magnetogram", database, args, lambda x: get_image(x, args.size), need_save=True)

    # AIA131
    if args.aia131:
        print("Prepare aia131 images ... ")
        _ = make_dataset("aia131", database, args, lambda x: get_image(x, args.size), need_save=True)

    # AIA1600
    if args.aia1600:
        print("Prepare aia1600 images ... ")
        _ = make_dataset("aia1600", database, args, lambda x: get_image(x, args.size), need_save=True)

    # Feature
    if args.physical:
        print("Prepare physical features ... ")
        datasets = make_dataset("feature", database, args)

        for year, dataset in datasets.items():
            prefix = make_prefix(output_path, year, "feat")
            with open(f"{prefix}.csv", "w") as wf:
                for sample in dataset:
                    wf.write(f"{sample}\n")

    # Label
    if args.label:
        print("Prepare labels ... ")
        datasets = make_dataset("flag", database, args)
        for year, dataset in datasets.items():
            prefix = make_prefix(output_path, year, "label")
            with open(f"{prefix}.csv", "w") as wf:
                for sample in dataset:
                    wf.write(f"{sample}\n")

    # Window index
    if args.window:
        print("Prepare window index ... ")
        idx = 0
        datasets = make_dataset("time", database, args)
        accm_times = []
        for year, times in tqdm(datasets.items()):
            prefix = f"{output_path}/data_{year}_window_{args.horizon}"
            accm_times.extend(times)
            with open(f"{prefix}.csv", "w") as wf:
                for time in times:
                    window = np.array(get_window(idx, accm_times, time, args.horizon)) - sections[year].start
                    window = [max(x, 0) for x in window]
                    window_str = ",".join(map(str, window))
                    wf.write(f"{window_str}\n")
                    idx += 1


if __name__ == "__main__":
    main()
