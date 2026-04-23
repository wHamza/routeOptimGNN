import os
import time
from collections import OrderedDict, namedtuple
from itertools import product

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from VRP_Actor import Model
from c import creat_data, reward1
from rolloutBaseline1 import RolloutBaseline
from runtime_utils import effective_batch_size, get_num_depots, get_runtime_device

device = get_runtime_device()
n_nodes = 52
steps = n_nodes * 2
num_depots = 1#get_num_depots(n_nodes)
max_grad_norm = 2


def rollout(model, dataset, batch_size, n_nodes):
    del batch_size
    model.eval()
    total_cost = []

    with torch.inference_mode():
        for bat in dataset:
            batch = bat.to(device)
            tour, _ = model(batch, n_nodes * 2, num_depots, True)
            total_cost.append(reward1(batch.x, tour.detach(), n_nodes).cpu())

    return torch.cat(total_cost, 0)


def adv_normalize(adv):
    std = adv.std()
    if torch.isnan(std) or std.item() == 0:
        return adv - adv.mean()
    return (adv - adv.mean()) / (std + 1e-8)


def train():
    class RunBuilder:
        @staticmethod
        def get_runs(params):
            run_type = namedtuple("Run", params.keys())
            return [run_type(*values) for values in product(*params.values())]

    params = OrderedDict(
        lr=[1e-4],
        batch_size=[64],
        hidden_node_dim=[128],
        hidden_edge_dim=[16],
        conv_laysers=[4],
        data_size=[128],
    )

    folder = f"Vrp-{n_nodes}-GAT"
    filename = "rollout"
    os.makedirs(os.path.join(folder, filename), exist_ok=True)

    for run in RunBuilder.get_runs(params):
        train_batch_size = effective_batch_size(run.batch_size, device, gpu_cap=128)
        valid_batch_size = effective_batch_size(run.batch_size, device, gpu_cap=64)

        print(
            "lr",
            "batch_size",
            "hidden_node_dim",
            "hidden_edge_dim",
            "conv_laysers:",
            run.lr,
            train_batch_size,
            run.hidden_node_dim,
            run.hidden_edge_dim,
            run.conv_laysers,
        )

        data_loder = creat_data(n_nodes, run.data_size, batch_size=train_batch_size)
        valid_loder = creat_data(n_nodes, 128, batch_size=valid_batch_size)
        print("Data creation completed")

        actor = Model(3, run.hidden_node_dim, 1, run.hidden_edge_dim, conv_laysers=run.conv_laysers).to(device)
        rol_baseline = RolloutBaseline(actor, valid_loder, n_nodes=n_nodes)
        actor_optim = optim.Adam(actor.parameters(), lr=run.lr)
        scheduler = LambdaLR(actor_optim, lr_lambda=lambda epoch_idx: 0.96 ** epoch_idx)

        costs = []
        for epoch in range(10):
            print("epoch:", epoch, "-------")
            actor.train()

            times, losses, rewards = [], [], []
            epoch_start = time.time()
            start = epoch_start

            for batch_idx, batch in enumerate(data_loder):
                batch = batch.to(device)
                tour_indices, tour_logp = actor(batch, steps, num_depots)

                reward_value = reward1(batch.x, tour_indices.detach(), n_nodes)
                base_reward = rol_baseline.eval(batch)

                advantage = adv_normalize(reward_value - base_reward)
                actor_loss = torch.mean(advantage.detach() * tour_logp)

                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()

                rewards.append(torch.mean(reward_value.detach()).item())
                losses.append(torch.mean(actor_loss.detach()).item())

                step_window = 200
                if (batch_idx + 1) % step_window == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end

                    mean_loss = np.mean(losses[-step_window:])
                    mean_reward = np.mean(rewards[-step_window:])
                    print(
                        "  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs"
                        % (batch_idx, len(data_loder), mean_reward, mean_loss, times[-1])
                    )

            scheduler.step()
            rol_baseline.epoch_callback(actor, epoch)

            epoch_dir = os.path.join(folder, filename, f"{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            save_path = os.path.join(epoch_dir, "actor.pt")
            torch.save(actor.state_dict(), save_path)

            cost = rollout(actor, valid_loder, valid_batch_size, n_nodes).mean()
            costs.append(cost.item())

            print("Problem:TSP%s / Average distance:" % n_nodes, cost.item())
            print(costs)