# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import argparse
import ast
import json
import logging
import multiprocessing as mp
import os
import random
import yaml
from functools import partial
from glob import glob
from typing import List, Tuple

import h5py
import MDAnalysis as mda
import numpy as np
import tqdm
from icecream import ic

from mace import data, tools
from mace.data.utils import save_configurations_as_HDF5
from mace.modules import compute_statistics
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_atomic_energies, get_dataset_from_mda, get_mda_universes
from mace.tools.utils import AtomicNumberTable


def compute_stats_target(
    file: str,
    z_table: AtomicNumberTable,
    r_max: float,
    atomic_energies: Tuple,
    batch_size: int,
    mda_universe: mda.Universe,
):
    train_dataset = data.HDF5Dataset(file, z_table=z_table, r_max=r_max, mda_universe=mda_universe)
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    avg_num_neighbors, mean, std = compute_statistics(train_loader, atomic_energies)
    output = [avg_num_neighbors, mean, std]
    return output


def pool_compute_stats(inputs: List):
    path_to_files, z_table, r_max, atomic_energies, mda_universe, batch_size, num_process = inputs

    with mp.Pool(processes=num_process) as pool:
        re = [
            pool.apply_async(
                compute_stats_target,
                args=(
                    file,
                    z_table,
                    r_max,
                    atomic_energies,
                    batch_size,
                    mda_universe,
                ),
            )
            for file in glob(path_to_files + "/*")
        ]

        pool.close()
        pool.join()

    results = [r.get() for r in tqdm.tqdm(re)]

    if not results:
        raise ValueError(
            "No results were computed. Check if the input files exist and are readable."
        )

    # Separate avg_num_neighbors, mean, and std
    avg_num_neighbors = np.mean([r[0] for r in results])
    means = np.array([r[1] for r in results])
    stds = np.array([r[2] for r in results])

    # Compute averages
    mean = np.mean(means, axis=0).item()
    std = np.mean(stds, axis=0).item()

    return avg_num_neighbors, mean, std


def split_array(a: np.ndarray, max_size: int):
    drop_last = False
    if len(a) % 2 == 1:
        a = np.append(a, a[-1])
        drop_last = True
    factors = get_prime_factors(len(a))
    max_factor = 1
    for i in range(1, len(factors) + 1):
        for j in range(0, len(factors) - i + 1):
            if np.prod(factors[j : j + i]) <= max_size:
                test = np.prod(factors[j : j + i])
                max_factor = max(test, max_factor)
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


# Define Task for Multiprocessiing
def multi_train_hdf5(process, file_paths, train_configs, drop_last):
    with h5py.File(f"train/{file_paths[process]}", "w") as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(train_configs[process], process, f)


def multi_valid_hdf5(process, file_paths, valid_configs, drop_last):
    with h5py.File(f"val/{file_paths[process]}", "w") as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(valid_configs[process], process, f)


def multi_test_hdf5(process, file_paths, test_configs, drop_last):
    with h5py.File(f"test/{file_paths[process]}", "w") as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(test_configs[process], process, f)

def weighted_mean(data, n_strucs):
    return np.sum([x * n for x, n in zip(data, n_strucs)]) / np.sum(n_strucs)


def main() -> None:
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    args = tools.build_preprocess_arg_parser().parse_args()
    run(args)


def run(args: argparse.Namespace):
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """

    # Setup
    tools.set_seeds(args.seed)
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    folders = ["train", "val", "test"]
    for sub_dir in folders:
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    # Data preparation
    with open(args.mda_universes, "r") as f:
        args.mda_universes, hdf5_files = get_mda_universes(yaml.safe_load(f))
    if hdf5_files["valid"] is None and "valid_fraction" in args:
        hdf5_files["valid"] = hdf5_files["train"]
    collections = get_dataset_from_mda(
        work_dir=args.work_dir,
        train_universes=args.mda_universes['train'],
        valid_universes=args.mda_universes['valid'],
        valid_fraction=args.valid_fraction,
        test_universes=args.mda_universes['test'],
        concatenate=False,
        seed=args.seed,
    )

    residues = np.array([mda_universe.residues.resnames for mda_universe in args.mda_universes["train"]])
    residues = np.unique(residues.flatten())
    z_table = AtomicNumberTable(list(range(np.unique(residues).shape[0]))) # Create fake z table
    E0s = ",".join([f"{z:d}: 0.0" for z in z_table.zs])
    E0s = "{" + E0s + "}"
    atomic_energies_dict = get_atomic_energies(E0s, None, z_table)

    logging.info("Preparing training set")
    if args.shuffle:
        for configs in collections.train:
            random.shuffle(configs)
    drop_last = False

    multi_train_hdf5_ = partial(multi_train_hdf5, file_paths=hdf5_files["train"], train_configs=collections.train, drop_last=drop_last)
    processes = []

    process_queue = [list(range(len(collections.train)))[i:i + args.num_process] for i in range(0, len(collections.train), args.num_process)]
    for process_batch in process_queue:
        for i in process_batch:
            p = mp.Process(target=multi_train_hdf5_, args=[i])
            p.start()
            processes.append(p)

        for i in processes:
            i.join()
    
    if args.compute_statistics:
        logging.info("Computing statistics")
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic Energies: {atomic_energies.tolist()}")
        avgs_num_neighbors = []
        means = []
        stds = []
        n_strucs = []
        for mda_universe in args.mda_universes["train"]:
            _inputs = [args.h5_prefix+'train', z_table, args.r_max, atomic_energies, mda_universe, args.batch_size, args.num_process]
            avg_num_neighbors, mean, std=pool_compute_stats(_inputs)
            avgs_num_neighbors.append(avg_num_neighbors)
            means.append(mean)
            stds.append(std)
            n_strucs.append(len(mda_universe.trajectory))
            logging.info(f"Statistics for universe {mda_universe}")
            logging.info(f"Average number of neighbors: {avg_num_neighbors}")
            logging.info(f"Mean: {mean}")
            logging.info(f"Standard deviation: {std}")

        # save the statistics as a json
        statistics = {
            "atomic_energies": str(atomic_energies_dict),
            "avg_num_neighbors": weighted_mean(avgs_num_neighbors, n_strucs),
            "mean": weighted_mean(means, n_strucs),
            "std": weighted_mean(stds, n_strucs),
            "atomic_numbers": str(z_table.zs),
            "r_max": args.r_max,
        }

        with open(args.h5_prefix + "statistics.json", "w") as f: # pylint: disable=W1514
            json.dump(statistics, f)

    logging.info("Preparing validation set")
    if args.shuffle:
        for configs in collections.valid:
            random.shuffle(configs)
    drop_last = False

    multi_valid_hdf5_ = partial(multi_valid_hdf5, file_paths=hdf5_files["valid"], valid_configs=collections.valid, drop_last=drop_last)
    processes = []
    
    process_queue = [list(range(len(collections.valid)))[i:i + args.num_process] for i in range(0, len(collections.valid), args.num_process)]
    for process_batch in process_queue:
        for i in process_batch:
            p = mp.Process(target=multi_valid_hdf5_, args=[i])
            p.start()
            processes.append(p)

        for i in processes:
            i.join()

    if args.mda_universes["test"] is not None:
        logging.info("Preparing test sets")
        
        drop_last = False

        multi_test_hdf5_ = partial(multi_test_hdf5, file_paths=hdf5_files["test"], test_configs=collections.test, drop_last=drop_last)
        processes = []

        process_queue = [list(range(len(collections.test)))[i:i + args.num_process] for i in range(0, len(collections.test), args.num_process)]
        for process_batch in process_queue:
            for i in process_batch:
                p = mp.Process(target=multi_test_hdf5_, args=[i])
                p.start()
                processes.append(p)

            for i in processes:
                i.join()
        

if __name__ == "__main__":
    main()
