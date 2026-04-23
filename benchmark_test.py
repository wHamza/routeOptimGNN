import os
import time

import torch

from VRP_Actor import Model
from creat_vrp import reward
from readdata import creat_data
from runtime_utils import get_runtime_device, load_model_state

device = get_runtime_device()


def rollout(model, dataset, n_nodes, greedy, nodes1, temperature, num_depots):
    model.eval()
    costs = []
    final_tour = None

    with torch.inference_mode():
        for bat in dataset:
            batch = bat.to(device)
            tour, _ = model(batch, n_nodes * 2, num_depots, greedy, temperature)
            costs.append(reward(nodes1, tour.detach(), n_nodes, num_depots).cpu())
            final_tour = tour

    return torch.cat(costs, 0), final_tour


def evaliuate(path, temperature=1, greedy=True):
    dl, nodes1, n_nodes, demand, capacity, dl1, num_depots = creat_data(path)
    del demand, capacity, dl1

    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
    min_cost = float("inf")
    best_tour = None
    elapsed = 0.0
    checkpoint_root = os.path.join(f"Vrp-{n_nodes}-GAT", "rollout")

    for epoch in range(84):
        checkpoint_path = os.path.join(checkpoint_root, f"{epoch}", "actor.pt")
        if not os.path.exists(checkpoint_path):
            continue

        load_model_state(agent, checkpoint_path, device)
        start_time = time.time()
        cost, tour = rollout(agent, dl, n_nodes, greedy, nodes1, temperature, num_depots)
        elapsed = time.time() - start_time

        scalar_cost = float(cost.min().item())
        if scalar_cost < min_cost:
            min_cost = scalar_cost
            best_tour = tour.detach().cpu().numpy()

        if greedy:
            print(cost, epoch, "----Gap", 100 * (cost - 6.1) / 6.1)
        else:
            print(cost, epoch, "Sampleing----Gap", "---------", temperature, 100 * (cost - 6.1) / 6.1)

    print(min_cost)
    return min_cost, elapsed, best_tour


def get_imlist(path):
    return [os.path.join(path, filename) for filename in os.listdir(path)]


if __name__ == "__main__":
    filename = "./data1"
    results = []
    times = []
    for path in get_imlist(filename):
        result, time1, _ = evaliuate(path)
        results.append(result)
        times.append(time1)
    print(results, times)
