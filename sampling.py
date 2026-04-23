import os
import time

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from VRP_Actor import Model
from creat_vrp import creat_instance, reward1
from runtime_utils import effective_batch_size, get_num_depots, get_runtime_device, load_model_state

n_nodes = 102
num_depots = get_num_depots(n_nodes)
device = get_runtime_device()


def rollout(model, dataset, temperature=1, greedy=False):
    model.eval()
    costs = []

    with torch.inference_mode():
        for bat in dataset:
            batch = bat.to(device)
            tour, _ = model(batch, n_nodes * 2, num_depots, greedy, temperature)
            costs.append(reward1(batch.x, tour.detach(), n_nodes).cpu())

    return torch.cat(costs, 0)


def evaliuate(valid_loder, temperature=1, greedy=False, checkpoint_epoch=69):
    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)

    checkpoint_path = os.path.join(f"Vrp-{n_nodes}-GAT", "rollout", str(checkpoint_epoch), "actor.pt")
    if os.path.exists(checkpoint_path):
        load_model_state(agent, checkpoint_path, device)

    start = time.time()
    cost = rollout(agent, valid_loder, temperature, greedy)
    elapsed = time.time() - start

    mean_cost = cost.mean()
    min_cost = cost.min()
    if greedy:
        print(mean_cost, "greedy----Gap", 100 * (mean_cost - 6.1) / 6.1)
    else:
        print(min_cost, "Sampleing----Gap", "---------", temperature, 100 * (min_cost - 6.1) / 6.1)

    print("Sampling took {:.2f}s".format(elapsed))
    return mean_cost, min_cost


def build_edge_index():
    node_ids = torch.arange(n_nodes, dtype=torch.long)
    return torch.cartesian_prod(node_ids, node_ids).t().contiguous()


def build_loader(node, edge, demand, capcity, repeats, batch_size):
    edge_index = build_edge_index()
    dataset = []
    for _ in range(repeats):
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


def load_arrays(data_size):
    csv_files = {
        "node": "vrp100_test_data.csv",
        "edge": "vrp100_distance.csv",
        "demand": "vrp100_demand.csv",
        "capcity": "vrp100_capcity.csv",
    }
    if all(os.path.exists(path) for path in csv_files.values()):
        node_ = np.loadtxt(csv_files["node"], dtype=float, delimiter=",")
        edges_ = np.loadtxt(csv_files["edge"], dtype=float, delimiter=",")
        demand_ = np.loadtxt(csv_files["demand"], dtype=float, delimiter=",")
        capcity_ = np.loadtxt(csv_files["capcity"], dtype=float, delimiter=",")
        instance_count = node_.size // (n_nodes * 2)
        return (
            node_.reshape(instance_count, n_nodes, 2),
            edges_.reshape(instance_count, -1, 1),
            demand_.reshape(instance_count, n_nodes),
            capcity_.reshape(-1),
        )

    generated_nodes = []
    generated_edges = []
    generated_demands = []
    generated_capcities = []
    for _ in range(data_size):
        data, edge, demand, capcity = creat_instance(0, n_nodes)
        generated_nodes.append(data)
        generated_edges.append(edge)
        generated_demands.append(demand)
        generated_capcities.append(capcity)

    return (
        np.array(generated_nodes),
        np.array(generated_edges),
        np.array(generated_demands),
        np.array(generated_capcities),
    )


def main():
    batch_size = effective_batch_size(1280, device, gpu_cap=64)
    data_size = 10000
    node_, edges_, demand_, capcity_ = load_arrays(data_size)

    all_scores = []
    for temperature in [2.5]:
        sample_costs = []
        for idx in range(min(1000, len(node_))):
            loader = build_loader(
                node_[idx],
                edges_[idx],
                demand_[idx],
                float(capcity_[idx]),
                repeats=batch_size,
                batch_size=batch_size,
            )
            _, sample_cost = evaliuate(loader, temperature, greedy=False)
            sample_costs.append(float(sample_cost))
            if idx % 100 == 0:
                print(sample_cost)

        all_scores.append(float(np.mean(sample_costs)))
        print(all_scores)
        np.savetxt("pn_1", np.array(all_scores), fmt="%f", delimiter=",")


if __name__ == "__main__":
    main()
