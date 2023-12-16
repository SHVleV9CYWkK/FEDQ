import argparse
import os
import time
import random

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset',
        help='name of dataset',
        type=str
    )

    parser.add_argument(
        '--n_clients',
        help='number of clients;',
        type=int,
        required=True
    )

    parser.add_argument(
        '--n_components',
        help='number of components/clusters; default is -1',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--is_split',
        help='Whether to split',
        type=bool,
        default=True)

    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar; '
             'default is 0.2',
        type=float,
        default=0.2)

    parser.add_argument(
        '--seed',
        help='Random Seed default is 54',
        type=int,
        default=54)

    parser.add_argument(
        '--flip_rate',
        help='probability of random reversal, default is 0.1',
        type=float,
        default=0.1)

    parser.add_argument(
        '--noise_level',
        help='Gaussian noise intensity, default is 0.1',
        type=float,
        default=0.1)

    return parser.parse_args()


def read_dataset(path):
    print("Reading raw data")
    files = [file for file in os.listdir(path) if file.endswith('.csv')]
    all_csv_list = os.listdir(path)
    all_data = None
    for csv_file in files:
        # cols = list(pd.read_csv(os.path.join(file_dir, csv_file), nrows=1, header=None))
        data = pd.read_csv(os.path.join(path, csv_file), header=None)
        if csv_file == all_csv_list[0]:
            all_data = data
        else:
            all_data = pd.concat([all_data, data], ignore_index=True)
    return all_data


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_dataset_by_labels(dataset, n_classes, n_clients, n_clusters, alpha, frac=1, seed=1234):
    """
    Split classification dataset among `n_clients`.
    ...
    :param dataset: numpy ndarray or pandas DataFrame where last column is label
    ...
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    print("Generating non-IID mixture distributions")
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}

    # Determine label array based on input type
    if isinstance(dataset, pd.DataFrame):
        labels = dataset.iloc[:, -1].values
    else:  # assume numpy
        labels = dataset[:, -1]

    for idx in selected_indices:
        label = labels[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

    return clients_indices


def ip_to_int(ip_str):
    octets = ip_str.split('.')
    return sum(int(octet) * (256 ** (3 - index)) for index, octet in enumerate(octets))


def check_clients_have_data(clients_data_indices):
    all_clients_have_data = True
    for i, client_indices in enumerate(clients_data_indices):
        if len(client_indices) < 3:
            print(f"Client {i} has no data!")
            all_clients_have_data = False

    if all_clients_have_data:
        print("All clients have been allocated data!")
    else:
        print("Some clients have no data allocated.")
    return all_clients_have_data
