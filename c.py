import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from runtime_utils import get_num_depots


CAPACITIES = {
    10: 2.0,
    20: 3.0,
    49: 25.0,
    50: 4.0,
    51: 12.0,
    99: 5.0,
    100: 5.0,
    101: 5.0,
    105: 5.0

}


def creat_instance(num, n_nodes=100, random_seed=None):
    del num
    rng = np.random.default_rng(random_seed)

    node_coords = rng.uniform(0, 1, (n_nodes, 2)).astype(np.float32)
    edge_vectors = node_coords[:, None, :] - node_coords[None, :, :]
    edges = np.linalg.norm(edge_vectors, axis=-1, keepdims=True).astype(np.float32)
    edges = edges.reshape(-1, 1)

    num_depots = 1#get_num_depots(n_nodes)
    customer_count = n_nodes - num_depots
    demand = rng.integers(1, 10, size=customer_count).astype(np.float32) / 10.0
    demand = np.concatenate(
        [np.zeros(num_depots, dtype=np.float32), demand],
        axis=0,
    )

    capcity = float(CAPACITIES[customer_count])
    return node_coords, edges, demand, capcity


def creat_data(n_nodes, num_samples=10000, batch_size=32):
    node_ids = torch.arange(n_nodes, dtype=torch.long)
    edge_index = torch.cartesian_prod(node_ids, node_ids).t().contiguous()

    dataset = []
    for _ in range(num_samples):
        node, edge, demand, capcity = creat_instance(num_samples, n_nodes)
        dataset.append(
            Data(
                x=torch.from_numpy(node).float(),
                edge_index=edge_index,
                edge_attr=torch.from_numpy(edge).float(),
                demand=torch.from_numpy(demand).unsqueeze(-1).float(),
                capcity=torch.tensor([capcity], dtype=torch.float32),
            )
        )

    return DataLoader(dataset, batch_size=batch_size)


def _as_coordinate_tensor(static, n_nodes, device=None):
    if isinstance(static, np.ndarray):
        static = torch.from_numpy(static).float()
    elif not torch.is_tensor(static):
        static = torch.tensor(static, dtype=torch.float32)

    static = static.reshape(-1, n_nodes, 2)
    if device is not None:
        static = static.to(device)
    return static


def _tour_coordinates(static, tour_indices):
    coords = static.transpose(1, 2)
    idx = tour_indices.long().unsqueeze(1).expand(-1, coords.size(1), -1)
    return torch.gather(coords, 2, idx).transpose(1, 2)


def _tour_length(tour):
    if tour.size(1) < 2:
        return torch.zeros(tour.size(0), device=tour.device)
    return torch.linalg.vector_norm(tour[:, 1:] - tour[:, :-1], dim=-1).sum(dim=1)


def _endpoint_costs(static, tour, num_depots):
    depots = static[:, :num_depots, :]
    start_cost = torch.cdist(tour[:, :1, :], depots).squeeze(1).min(dim=-1).values
    end_cost = torch.cdist(tour[:, -1:, :], depots).squeeze(1).min(dim=-1).values
    return start_cost + end_cost


def _reward_core(static, tour_indices, n_nodes, num_depots):
    static = _as_coordinate_tensor(static, n_nodes, device=tour_indices.device)
    tour = _tour_coordinates(static, tour_indices)
    total_length = _tour_length(tour) + _endpoint_costs(static, tour, num_depots)
    return total_length.detach()


def reward(static, tour_indices, n_nodes, num_depotss):
    return _reward_core(static, tour_indices, n_nodes, num_depotss)


def reward2(static, tour_indices, n_nodes, num_depotss):
    return _reward_core(static, tour_indices, n_nodes, num_depotss)


def reward1(static, tour_indices, n_nodes):
    return _reward_core(static, tour_indices, n_nodes, get_num_depots(n_nodes))
