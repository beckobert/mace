###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import ast
import json
import logging
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch.nn.functional
from e3nn.util import jit
from icecream import ic
from torch.utils.data import ConcatDataset
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, tools
from mace.modules import coarse_graining
from mace.tools import torch_geometric
from mace.tools.model_script_utils import configure_model
from mace.tools.multihead_tools import (
    HeadConfig,
    dict_head_to_dataclass,
    prepare_default_head,
)
from mace.tools.scripts_utils import (
    LRScheduler,
    convert_to_json_format,
    create_error_table,
    dict_to_array,
    extract_config_mace_model,
    get_atomic_energies,
    get_avg_num_neighbors,
    get_dataset_from_mda,
    get_loss_fn,
    get_mda_universes,
    get_optimizer,
    get_params_options,
    print_git_commit,
    setup_wandb,
)
from mace.tools.utils import AtomicNumberTable


def main() -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args = tools.build_default_arg_parser().parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    """
    This script runs the training/fine tuning for mace
    """
    tag = tools.get_tag(name=args.name, seed=args.seed)
    args, input_log_messages = tools.check_args(args)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=0)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    commit = print_git_commit()
    args.multiheads_finetuning = False

    if args.mda_universes is not None:
        args.mda_universes = get_mda_universes(args)
        if args.biasing_potential is not None:
            bias_potentials = coarse_graining.read_biasing_potential(args.biasing_potential)
            for u_name, bias_pots in bias_potentials.items():
                for mda_universe_set in args.mda_universes.values():
                    if isinstance(mda_universe_set, Dict) and u_name in mda_universe_set.keys():
                        mda_universe_set[u_name]["bias_potential"] = bias_pots 
    for arg in [args.train_file, args.valid_file, args.test_file]:
        if arg is not None:
            logging.warning(
                'CG mode only supports giving datasets through mda-universes and directories to HDF5 files'
            )
    if args.heads is not None:
        args.heads = ast.literal_eval(args.heads)
    else:
        args.heads = prepare_default_head(args)

    logging.info("===========LOADING INPUT DATA===========")
    
    heads = list(args.heads.keys())
    residues = None
    logging.info(f"Using heads: {heads}")

    head_configs: List[HeadConfig] = []
    for head, head_args in args.heads.items():
        logging.info(f"=============    Processing head {head}     ===========")
        head_config = dict_head_to_dataclass(head_args, head, args)
        if head_config.statistics_file is not None:
            with open(head_config.statistics_file, "r") as f:  # pylint: disable=W1514
                statistics = json.load(f)
            logging.info("Using statistics json file")
            head_config.r_max = (
                statistics["r_max"] if args.foundation_model is None else args.r_max
            )
            head_config.atomic_numbers = statistics["atomic_numbers"]
            head_config.mean = statistics["mean"]
            head_config.std = statistics["std"]
            head_config.avg_num_neighbors = statistics["avg_num_neighbors"]
            head_config.compute_avg_num_neighbors = False
            residues = np.array(statistics["model_residues"])
            if isinstance(statistics["atomic_energies"], str) and statistics[
                "atomic_energies"
            ].endswith(".json"):
                with open(statistics["atomic_energies"], "r", encoding="utf-8") as f:
                    atomic_energies = json.load(f)
                head_config.E0s = atomic_energies
                head_config.atomic_energies_dict = ast.literal_eval(atomic_energies)
            else:
                head_config.E0s = statistics["atomic_energies"]
                head_config.atomic_energies_dict = ast.literal_eval(
                    statistics["atomic_energies"]
                )
        head_configs.append(head_config)

    logging.info(f"No Atomic Numbers/Energies due to coarse graining mode.")
    if residues is None:
        residues = []
        for head_config in head_configs:
            for mda_universe in head_config.mda_universes["train"].values():
                residues.append(mda_universe["universe"].residues.resnames)
        residues = np.unique(np.concatenate(residues))
    z_table = AtomicNumberTable(list(range(residues.shape[0]))) # Create fake z table
    E0s = ",".join([f"{z:d}: 0.0" for z in z_table.zs])
    E0s = "{" + E0s + "}"
    # This creates an atomic energy dict where zs from z_table are missing. But since E0s are useless in CG, we don't care.
    # This dict is just a formality
    atomic_energies_dict = {i: 0.0 for i in range(residues.shape[0])}

    for head_config in head_configs:
        # Data preparation
        if not args.mda_universes["hdf5"]:
            collections = get_dataset_from_mda(
                args=args,
                work_dir=args.work_dir,
                train_universes=head_config.mda_universes["train"],
                valid_universes=head_config.mda_universes["valid"],
                valid_fraction=head_config.valid_fraction,
                test_universes=head_config.mda_universes["test"],
                model_residues=residues,
                seed=args.seed,
                head_name=head_config.head_name,
            )

            head_config.collections = collections

    # Atomic number table
    # yapf: disable
    dipole_only = False
    args.compute_dipole = False
    atomic_energies = dict_to_array(atomic_energies_dict, heads)
    args.compute_energy = False
    args.loss_function = "forces_only"
    args.scaling = "no_scaling"

    valid_sets = {head: [] for head in heads}
    train_sets = {head: [] for head in heads}
    for head_config in head_configs:
        if not args.mda_universes["hdf5"]:
            train_sets[head_config.head_name] = [
                data.AtomicData.from_mda_config(
                    config, cutoff=args.r_max, heads=heads,
                )
                for config in head_config.collections.train
            ]
            valid_sets[head_config.head_name] = [
                data.AtomicData.from_mda_config(
                    config, cutoff=args.r_max, heads=heads,
                )
                for config in head_config.collections.valid
            ]
        # elif head_config.train_file.endswith(".h5"):
        #     train_sets[head_config.head_name] = data.HDF5Dataset(
        #         head_config.train_file, r_max=args.r_max, z_table=z_table, 
        #         heads=heads, head=head_config.head_name, mda_universe=head_config.mda_universes['train']
        #     )
        #     valid_sets[head_config.head_name] = data.HDF5Dataset(
        #         head_config.valid_file, r_max=args.r_max, z_table=z_table,
        #         heads=heads, head=head_config.head_name, mda_universe=head_config.mda_universes['valid']
        #     )
        else:  # This case would be for when the file path is to a directory of multiple .h5 files
            train_sets[head_config.head_name] = data.combine_hdf5_datasets(
                files=[f"train/{fn}" for fn in [f"{name}.h5" for name in args.mda_universes["train"].keys()]], r_max=args.r_max, z_table=z_table,
                heads=heads, head=head_config.head_name, mda_universes=head_config.mda_universes["train"]
            )
            valid_sets[head_config.head_name] = data.combine_hdf5_datasets(
                files=[f"val/{fn}" for fn in [f"{name}.h5" for name in args.mda_universes["valid"].keys()]], r_max=args.r_max, z_table=z_table,
                heads=heads, head=head_config.head_name, mda_universes=head_config.mda_universes["valid"]
            )
            
            
        train_loader_head = torch_geometric.dataloader.DataLoader(
            dataset=train_sets[head_config.head_name],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        head_config.train_loader = train_loader_head
    # concatenate all the trainsets
    train_set = ConcatDataset([train_sets[head] for head in heads])
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loaders = {heads[i]: None for i in range(len(heads))}
    if not isinstance(valid_sets, dict):
        valid_sets = {"Default": valid_sets}
    for head, valid_set in valid_sets.items():
        valid_loaders[head] = torch_geometric.dataloader.DataLoader(
            dataset=valid_set,
            batch_size=args.valid_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )

    loss_fn = get_loss_fn(args, dipole_only, args.compute_dipole)
    args.avg_num_neighbors = get_avg_num_neighbors(head_configs, args, train_loader, device)

    # Model
    model, output_args = configure_model(args, train_loader, atomic_energies, None, heads, z_table)
    # Add residues
    model.residues = residues
    model.to(device)

    logging.debug(model)
    logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
    logging.info("")
    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(
        f"Number of gradient updates: {int(args.max_num_epochs*len(train_set)/args.batch_size)}"
    )
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Optimizer
    param_options = get_params_options(args, model)
    optimizer: torch.optim.Optimizer
    optimizer = get_optimizer(args, param_options)
    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=None,
    )

    start_epoch = 0
    if args.restart_latest:
        opt_start_epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=False,
            device=device,
        )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    if args.wandb:
        setup_wandb(args)

    distributed_model = None

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=None,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        distributed=args.distributed,
        distributed_model=distributed_model,
        train_sampler=None,
        rank=0,
    )

    logging.info("")
    logging.info("===========RESULTS===========")
    logging.info("Computing metrics for training, validation, and test sets")

    train_valid_data_loader = {}
    for head_config in head_configs:
        data_loader_name = "train_" + head_config.head_name
        train_valid_data_loader[data_loader_name] = head_config.train_loader
    for head, valid_loader in valid_loaders.items():
        data_load_name = "valid_" + head
        train_valid_data_loader[data_load_name] = valid_loader

    test_sets = {}
    stop_first_test = False
    test_data_loader = {}
    if all(
        head_config.test_file == head_configs[0].test_file
        for head_config in head_configs
    ) and head_configs[0].test_file is not None:
        stop_first_test = True
    if all(
        head_config.test_dir == head_configs[0].test_dir
        for head_config in head_configs
    ) and head_configs[0].test_dir is not None:
        stop_first_test = True
    for head_config in head_configs:
        if args.mda_universes["hdf5"]:
            for name, mda_universe in zip(head_configs.mda_universes["test"].items()):
                test_sets[name] = data.HDF5Dataset(
                    f"test/{name}.h5", r_max=args.r_max, z_table=z_table, heads=heads,
                    head=head_config.head_name, mda_universe=mda_universe
                )
        for test_name, test_set in test_sets.items():
            print(test_name)
            test_sampler = None
            try:
                drop_last = test_set.drop_last
            except AttributeError as e:  # pylint: disable=W0612
                drop_last = False
            test_loader = torch_geometric.dataloader.DataLoader(
                test_set,
                batch_size=args.valid_batch_size,
                shuffle=(test_sampler is None),
                drop_last=drop_last,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            test_data_loader[test_name] = test_loader
        if stop_first_test:
            break

    epoch = checkpoint_handler.load_latest(
        state=tools.CheckpointState(model, optimizer, lr_scheduler),
        swa=False,
        device=device,
    )
    model.to(device)
    logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

    for param in model.parameters():
        param.requires_grad = False
    table_train_valid = create_error_table(
        table_type=args.error_table,
        all_data_loaders=train_valid_data_loader,
        model=model,
        loss_fn=loss_fn,
        output_args=output_args,
        log_wandb=args.wandb,
        device=device,
        distributed=args.distributed,
    )
    logging.info("Error-table on TRAIN and VALID:\n" + str(table_train_valid))

    if test_data_loader:
        table_test = create_error_table(
            table_type=args.error_table,
            all_data_loaders=test_data_loader,
            model=model,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
        )
        logging.info("Error-table on TEST:\n" + str(table_test))

    # Save entire model
    model_path = Path(args.checkpoints_dir) / (tag + ".model")
    logging.info(f"Saving model to {model_path}")
    if args.save_cpu:
        model = model.to("cpu")
    torch.save(model, model_path)
    extra_files = {
        "commit.txt": commit.encode("utf-8") if commit is not None else b"",
        "config.yaml": json.dumps(
            convert_to_json_format(extract_config_mace_model(model))
        ),
    }
    
    torch.save(model, Path(args.model_dir) / (args.name + ".model"))
    try:
        path_complied = Path(args.model_dir) / (
            args.name + "_compiled.model"
        )
        logging.info(f"Compiling model, saving metadata to {path_complied}")
        model_compiled = jit.compile(deepcopy(model))
        torch.jit.save(
            model_compiled,
            path_complied,
            _extra_files=extra_files,
        )
    except Exception as e:  # pylint: disable=W0703
        pass

    logging.info("Done")


if __name__ == "__main__":
    main()
