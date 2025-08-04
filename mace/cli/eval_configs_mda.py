###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.io
import MDAnalysis as mda
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils

kjmol = 0.01036427 # from kj/mol/A to eV/A

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", help="path to coordinates file", required=True)
    parser.add_argument("--topology", help="path to topology file", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help with CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    u = mda.Universe(args.topology, *[args.coordinates])
    configs = data.load_from_mda_universe(universe=u, residues=model.residues)

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_mda_config(
                config, universe=u, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        # ic(batch.to_dict())
        output = model(batch.to_dict())
        energies_list.append(torch_tools.to_numpy(output["energy"]))

        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its empty

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    assert len(u.trajectory) == len(energies) == len(forces_list)

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(u.trajectory) == contributions.shape[0]

    mace_universe = mda.Universe.empty(
        u.atoms.n_atoms,
        n_residues=u.residues.n_residues,
        atom_resindex=u.atoms.resindices,
        trajectory=True,
        forces=True    
    )

    print(energies)
    
    # Store data in atoms objects
    with mda.Writer(args.output, n_atoms=u.atoms.n_atoms) as writer:
        for i, (ts, energy, forces) in enumerate(zip(u.trajectory, energies, forces_list)):
            mace_universe.atoms.positions = u.atoms.positions
            mace_universe.atoms.forces = forces / kjmol
            writer.write(mace_universe)

if __name__ == "__main__":
    main()
