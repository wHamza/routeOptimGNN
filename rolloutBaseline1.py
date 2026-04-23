import copy

import torch
from scipy.stats import ttest_rel
from torch.nn import DataParallel

from creat_vrp import reward1
from runtime_utils import get_num_depots, get_runtime_device

device = get_runtime_device()


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def rollout(model, dataset, n_nodes):
    model.eval()
    num_depots = get_num_depots(n_nodes)
    costs = []

    with torch.inference_mode():
        for bat in dataset:
            batch = bat.to(device)
            tour, _ = model(batch, n_nodes * 2, num_depots, True)
            costs.append(reward1(batch.x, tour.detach(), n_nodes).cpu())

    return torch.cat(costs, 0)


class RolloutBaseline:
    def __init__(self, model, dataset, n_nodes=50, epoch=0):
        super(RolloutBaseline, self).__init__()
        self.n_nodes = n_nodes
        self.num_depots = get_num_depots(n_nodes)
        self.dataset = dataset
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model).to(device)
        self.bl_vals = rollout(self.model, self.dataset, n_nodes=self.n_nodes).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch
        if dataset is not None:
            self.dataset = dataset

    def eval(self, x, n_nodes=None):
        del n_nodes
        with torch.inference_mode():
            tour, _ = self.model(x, self.n_nodes * 2, self.num_depots, True)
            return reward1(x.x, tour.detach(), self.n_nodes)

    def epoch_callback(self, model, epoch):
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.n_nodes).cpu().numpy()
        candidate_mean = candidate_vals.mean()

        print(
            "Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
                epoch,
                candidate_mean,
                self.epoch,
                self.mean,
                candidate_mean - self.mean,
            )
        )
        if candidate_mean - self.mean < 0:
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < 0.05:
                print("Update baseline")
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            "model": self.model,
            "dataset": self.dataset,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict["model"]).state_dict())
        self._update_model(load_model, state_dict["epoch"], state_dict["dataset"])
