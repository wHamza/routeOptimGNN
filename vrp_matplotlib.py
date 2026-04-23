import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from VRP_Actor import Model
from creat_vrp import reward2
from runtime_utils import get_num_depots, get_runtime_device, load_model_state

device = get_runtime_device()


def split_routes(route, num_depots):
    routes = []
    current_route = [0]

    for raw_node in route:
        node = int(raw_node)
        if node < num_depots:
            current_route.append(node)
            if any(customer >= num_depots for customer in current_route):
                routes.append(np.array(current_route))
            current_route = [node]
        else:
            current_route.append(node)

    if any(customer >= num_depots for customer in current_route):
        current_route.append(current_route[0] if current_route[0] < num_depots else 0)
        routes.append(np.array(current_route))

    return routes


def plot_vehicle_routes(cost, data, route, ax1, greedy, num_depots, markersize=5):
    plt.rc("font", family="Times New Roman", size=10)

    route = route.detach().cpu().numpy()
    routes = split_routes(route, num_depots)
    locs = data.x.cpu().numpy()

    depot_coords = locs[:num_depots]
    customer_coords = locs[num_depots:]
    ax1.scatter(customer_coords[:, 0], customer_coords[:, 1], s=markersize * 10, c="black", alpha=0.7)
    ax1.scatter(depot_coords[:, 0], depot_coords[:, 1], s=markersize * 60, c="red", marker="s")

    colors = plt.cm.get_cmap("tab20", max(1, len(routes)))
    for route_idx, vehicle_route in enumerate(routes):
        coords = locs[vehicle_route]
        ax1.plot(coords[:, 0], coords[:, 1], marker="o", linewidth=1.8, color=colors(route_idx))

    title = "Greedy" if greedy else "Sampling"
    ax1.set_title(f"{title}; route_count={len(routes)}; total_distance={float(cost.item()):.2f}")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)


def infer_n_nodes(node_array, capacity_array):
    instance_count = max(1, capacity_array.reshape(-1).shape[0])
    total_values = node_array.size
    if total_values % (instance_count * 2) != 0:
        raise ValueError(
            f"Node data size {total_values} is not compatible with {instance_count} instances of 2D coordinates."
        )
    return total_values // (instance_count * 2)


def load_dataset_arrays(dataset_prefix):
    node_path = f"{dataset_prefix}_test_data.csv"
    demand_path = f"{dataset_prefix}_demand.csv"
    capcity_path = f"{dataset_prefix}_capcity.csv"

    node_ = np.loadtxt(node_path, dtype=float, delimiter=",")
    demand_ = np.loadtxt(demand_path, dtype=float, delimiter=",")
    capcity_ = np.loadtxt(capcity_path, dtype=float, delimiter=",")

    n_nodes = infer_n_nodes(node_, capcity_)
    node_ = node_.reshape(-1, n_nodes, 2)
    demand_ = demand_.reshape(-1, n_nodes)
    capcity_ = capcity_.reshape(-1)
    return node_, demand_, capcity_, n_nodes


def build_loader(dataset_prefix, instance_idx):
    node_, demand_, capcity_, n_nodes = load_dataset_arrays(dataset_prefix)

    edges = np.linalg.norm(node_[instance_idx][:, None, :] - node_[instance_idx][None, :, :], axis=-1, keepdims=True)
    edges_ = edges.reshape(-1, 1)

    node_ids = torch.arange(n_nodes, dtype=torch.long)
    edge_index = torch.cartesian_prod(node_ids, node_ids).t().contiguous()

    data = Data(
        x=torch.from_numpy(node_[instance_idx]).float(),
        edge_index=edge_index,
        edge_attr=torch.from_numpy(edges_).float(),
        demand=torch.from_numpy(demand_[instance_idx]).unsqueeze(-1).float(),
        capcity=torch.tensor([float(capcity_[instance_idx])], dtype=torch.float32),
    )
    return DataLoader([data], batch_size=1), n_nodes


def load_agent(n_nodes, checkpoint_epoch=51):
    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
    checkpoint_path = os.path.join(f"Vrp-{n_nodes}-GAT", "rollout", str(checkpoint_epoch), "actor.pt")
    if os.path.exists(checkpoint_path):
        load_model_state(agent, checkpoint_path, device)
    return agent


def vrp_matplotlib(greedy=True, dataset_prefix="vrp20", instance_idx=0, checkpoint_epoch=51, temperature=1.2, max_routes=None):
    data_loader, n_nodes = build_loader(dataset_prefix, instance_idx)
    num_depots = get_num_depots(n_nodes)
    agent = load_agent(n_nodes, checkpoint_epoch)
    batch = next(iter(data_loader)).to(device)

    with torch.inference_mode():
        tour, _ = agent(batch, n_nodes * 2, num_depots, greedy, temperature, max_routes=max_routes)
        cost = reward2(batch.x, tour.detach(), n_nodes, num_depots)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_vehicle_routes(cost, batch, tour[0], ax, greedy, num_depots)
    plt.savefig(f"./{dataset_prefix}-route-{instance_idx}.svg", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    vrp_matplotlib(greedy=True, dataset_prefix="vrp20", instance_idx=0, max_routes=1)
