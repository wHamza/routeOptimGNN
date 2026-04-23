import os
import time

import torch

from VRP_Actor import Model
from creat_vrp import creat_data, reward1
from runtime_utils import effective_batch_size, get_num_depots, get_runtime_device, load_model_state

device = get_runtime_device()
n_nodes = 22
num_depots = get_num_depots(n_nodes)


def rollout(model, dataset, temperature=1, greedy=False):
    model.eval()
    costs = []

    with torch.inference_mode():
        for bat in dataset:
            batch = bat.to(device)
            tour, _ = model(batch, n_nodes * 2, num_depots, greedy, temperature)
            costs.append(reward1(batch.x, tour.detach(), n_nodes).cpu())

    return torch.cat(costs, 0)


def evaliuate(valid_loder, temperature=1, greedy=False, checkpoint_epoch=22):
    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)

    checkpoint_dir = os.path.join(f"Vrp-{n_nodes}-GAT", "rollout", str(checkpoint_epoch))
    checkpoint_path = os.path.join(checkpoint_dir, "actor.pt")
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

    print("Evaluation took {:.2f}s".format(elapsed))
    return mean_cost, min_cost


if __name__ == "__main__":
    batch_size = effective_batch_size(256, device, gpu_cap=64)
    valuate_size = 10
    valid_loder = creat_data(n_nodes, valuate_size, batch_size=batch_size)
    evaliuate(valid_loder, temperature=1, greedy=True)
