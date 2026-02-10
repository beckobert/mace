# This file contains useful functions when running in coarse grain mode
# Note, that there other functions for coarse grain mode in other files as well

import ast
import itertools

import numpy as np
import numpy.linalg as la
import torch
from icecream import ic

from mace.data.neighborhood import get_neighborhood
from mace.data import Configuration
from mace.modules.utils import get_edge_vectors_and_lengths


k_B = 8.617333262e-5

def fit_quadratic(x, y):
    id_filter = [i for i, y_value in enumerate(y) if not np.isinf(y_value)]
    coefficients = np.polyfit(x[id_filter], y[id_filter], 2)
    return coefficients

def calculate_harmonic_coefficients(collection, bonds, nbins=20, k_B=k_B, T=300.0):
    # For now, have only bonds that allow only to load data from a single mda.Universe by using bonds[0]
    # TODO: This should be improved after proof of concept
    # Concatenate the collection, if necessary
    if isinstance(collection[0], list):
        collection = list(itertools.chain.from_iterable(collection))
    distances = np.zeros((len(collection), len(bonds[0])))
    cell_volumes = np.zeros((len(collection,)))
    bonds_0 = [bond[0] for bond in bonds[0]]
    bonds_1 = [bond[1] for bond in bonds[0]]
    for i, config in enumerate(collection):
        vector = config.positions[bonds_0] - config.positions[bonds_1]
        distances[i] = la.norm(vector, axis=-1)
        cell_volumes[i] = np.abs(la.det(config.cell))
    
    coefficients, minima = [], []
    for i, bonds in enumerate(bonds[0]):
        hist, edge = np.histogram(distances[:, i], bins=nbins)
        center = (edge[1:] + edge[:-1]) / 2
        
        V = (4 / 3) * np.pi * (np.power(edge[1:], 3) - np.power(edge[:-1], 3))
        norm = np.sum(1.0 / cell_volumes) * V
        hist_norm = hist / norm

        pot_mean_frc = -k_B * T * np.log(hist_norm)
        # Save calculation for future analysis and debugging
        np.savez(
            f'potential_of_mean_force_{i+1:d}.npz',
            pot_mean_frc=pot_mean_frc,
            center=center,
            hist_norm=hist_norm,
            edge=edge
        )

        coefficient = fit_quadratic(center, pot_mean_frc)
        coefficients.append(coefficient)
        minima.append(coefficient[1] / (-2 * coefficient[0]))

    return np.array(coefficients), np.array(minima)

def write_harmonic_coefficients(coefficients, minima, bonds, fn):
    # For now, have only bonds that allow only to load data from a single mda.Universe by using bonds[0]
    # TODO: This should be improved after proof of concept
    with open("harmonic_potential.data", 'w') as fn:
        fn.write("# Coefficients of harmonic potential a * x**2 + b * x + c\n")
        fn.write("# bead 1  bead 2       a       b       c minimum\n")
        for bond, coeff, minimum in zip(bonds[0], coefficients, minima):
            fn.write(f"{bond[0]:>8d} {bond[1]:>7d} {coeff[0]:>7.4f} {coeff[1]:>7.4f} {coeff[2]:>7.4f} {minimum:>7.3f}\n")


def read_biasing_potential(fn, training=True):
    with open(fn, 'r') as f:
        start_harm = None
        start_rep = None
        start_defs = None

        harm_pots = {}
        rep_pots = {}
        defined_pairs = {}
        
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Harmonic potentials" in line:
                if start_harm is not None:
                    raise ValueError("Two Harmonic potential keywords have been found")
                start_harm = i
            elif "Repulsive potentials" in line:
                if start_rep is not None:
                    raise ValueError("Two Repulsive potential keywords have been found")
                start_rep = i
            elif "Definition of pairs" in line:
                if start_defs is not None:
                    raise ValueError("Two Definition of pairs keywords have been found")
                start_defs = i

        for line in lines[start_harm:]:
            ls = line.split()
            if ls == []:
                break
            if line[0] != "#":
                harm_pots[ls[0]] = [float(ls[1]), float(ls[2]), float(ls[3])]

        for line in lines[start_rep:]:
            ls = line.split()
            if ls == []:
                break
            if line[0] != "#":
                rep_pots[ls[0]] = [float(ls[1]), float(ls[2])]
            

        universe = []
        if not training:
            u_name = "Calculator"
        for line in lines[start_defs:]:
            ls = line.split()
            if ls == []:
                defined_pairs[u_name] = universe
                universe = []
            elif "universe" in line:
                u_name = ls[1]
            elif line[0] != "#":
                if ls[2] in ["y", "t", "1"]:
                    universe.append([int(ls[0]), int(ls[1]), harm_pots[ls[3]]])
                else:
                    universe.append([int(ls[0]), int(ls[1]), rep_pots[ls[3]]])
        defined_pairs[u_name] = universe

    return defined_pairs        


def calculate_bias_potential(positions, edge_index, shifts, pairs, coefficients, bias_type):
    # Obtain a list of edge indices that match a specified bead pair
    matches = (edge_index.T.unsqueeze(0) == pairs.unsqueeze(1))
    all_matches = torch.nonzero(matches.all(dim=2))
    included_pairs = all_matches[:, 0].tolist()
    matched_indices = all_matches[:, 1].tolist()

    edge_index = edge_index[:, matched_indices]
    shifts = shifts[matched_indices]
    pairs = pairs[included_pairs]
    coefficients = coefficients[included_pairs]
    
    vectors, lengths = get_edge_vectors_and_lengths(positions=positions, edge_index=edge_index, shifts=shifts)

    if bias_type == "harmonic":
        energy_pairs = (
            coefficients[:, 0] * lengths**2
            + coefficients[:, 1] * lengths
            + coefficients[:, 2]
        )
        force_strength = (-2 * coefficients[:, 0] * lengths - coefficients[:, 1])
    elif bias_type == "repulsive":
        potential_active = (lengths <= 2**(1/6) * coefficients[:, 1])
        energy_pairs = (
            4 * coefficients[:, 0]
            * (torch.pow(coefficients[:, 1] / lengths, 12) - torch.pow(coefficients[:, 1] / lengths, 6))
            + coefficients[:, 0]
        )
        energy_pairs = energy_pairs * potential_active
        force_strength = (
            -4 * coefficients[:, 0]
            * (
                -12 * torch.pow(coefficients[:, 1], 12) * torch.pow(lengths, -13) 
                + 6 * torch.pow(coefficients[:, 1], 6) * torch.pow(lengths, -7)
            )
        )
        force_strength = force_strength * potential_active
    else:
        raise ValueError(f"Bias type must be harmonic or repulsive, instead is {bias_type}")
    force_pairs = force_strength * vectors / lengths
    # Calculate the potential energy and force added to each bead
    energy_nodes = torch.zeros(positions.shape[0], device=positions.device)
    forces = torch.zeros_like(positions)
    for pair, ener, frc in zip(pairs.tolist(), energy_pairs, force_pairs):
        energy_nodes[pair] += ener / 2
        forces[pair[0]] += frc
        forces[pair[1]] -= frc

    return energy_nodes, forces

def subtract_bias_potential(configs, bias_potential, args):
    pairs_harm = [[p[0], p[1]] for p in bias_potential if len(p[2]) == 3]
    coefficients_harm = [p[2] for p in bias_potential if len(p[2]) == 3]
    pairs_rep = [[p[0], p[1]] for p in bias_potential if len(p[2]) == 2]
    coefficients_rep = [p[2] for p in bias_potential if len(p[2]) == 2]
    # ic(pairs_harm)
    # ic(coefficients_harm)

    edge_index, shifts, _, _ = get_neighborhood(
        positions=config.positions,
        cutoff=args.cutoff,
        pbc=config.pbc,
        cell=config.cell,
    )
    # Transfering data to torch tensors
    positions = torch.tensor(config.positions)
    shifts = torch.tensor(shifts)
    edge_index = torch.tensor(edge_index, device=positions.device, dtype=torch.int)

    for config in configs:
        if len(pairs_harm) > 0:
            coefficients = torch.tensor(coefficients_harm, device=positions.device).unsqueeze(-1)
            pairs = torch.tensor(pairs_harm, device=edge_index.device, dtype=int)
            _, harm_forces = calculate_bias_potential(
                positions=positions,
                edge_index=edge_index,
                shifts=shifts,
                pairs=pairs,
                coefficients=coefficients,
                bias_type="harmonic",
            )
            config.forces += np.array(harm_forces)
        if len(pairs_rep) > 0:
            coefficients = torch.tensor(coefficients_rep, device=positions.device).unsqueeze(-1)
            pairs = torch.tensor(pairs_rep, device=edge_index.device, dtype=int)
            _, rep_forces = calculate_bias_potential(
                positions=positions,
                edge_index=edge_index,
                shifts=shifts,
                pairs=pairs,
                coefficients=coefficients,
                bias_type="repulsive",
            )
            config.forces += np.array(rep_forces)
        # ic.disable()
        
    return configs

def subtract_harm_from_collections(collection, **kwargs):
    if len(collection) == 0:
        return collection
    ic(collection == [])
    if isinstance(collection[0], Configuration):
        collection = subtract_harm_from_collection(collection, **kwargs)
    else:
        for coll in collection:
            coll = subtract_harm_from_collection(coll, **kwargs)

    # if isinstance(collection, list):
    #     for coll in collection:
    #         coll = subtract_harm_from_collection(coll, **kwargs)
    # else:
    #     collection = subtract_harm_from_collection(collection, **kwargs)
    return collection


def subtract_harm_from_collection(collection, cutoff, **kwargs):
    for config in collection:
        edge_index, shifts, _, _ = get_neighborhood(
            positions=config.positions,
            cutoff=cutoff,
            pbc=config.pbc,
            cell=config.cell,
        )
        bonds = []
        for index in kwargs["bonds"][0]:
            # edge index contains all edges in the graph within the cutoff radius
            # Compare both rows of edge_indices with bond_indices, then multiply the rows so there is only a 1 where both match
            # No need for permutations, we don't want doubble counting
            # np.nonzero gives the indice of non-zero values
            bonds.append(np.nonzero(np.prod(edge_index == np.array(index)[:, np.newaxis], axis=0))[0].item())
        if len(bonds) != len(kwargs["bonds"][0]):
            raise Warning("A specified bond is longer than the specified cutoff radius")
        _, harm_forces = calculate_harmonic_potential(
            positions=config.positions,
            shifts=shifts[bonds],
            **kwargs
        )
        config.forces += np.array(harm_forces)
    return collection

