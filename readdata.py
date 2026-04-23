import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data, DataLoader

from data_classes import Customer, Depot


def load_problem(path):
    capcity = 0
    depots = []
    demands = []
    x_y = []
    customers = []

    with open(path) as f:
        max_vehicles, num_customers, num_depots = map(int, f.readline().strip().split())

        for _ in range(num_depots):
            max_duration, max_load = map(int, f.readline().strip().split())
            depots.append(Depot(max_vehicles, max_duration, max_load))
            capcity = max_load

        for _ in range(num_customers):
            cid, x, y, service_duration, demand = map(int, f.readline().strip().split())
            customers.append(Customer(cid, x, y, service_duration, demand))
            x_y.append([x, y])
            demands.append(demand)

        for depot_idx in range(num_depots):
            cid, x, y = map(int, f.readline().strip().split())
            depots[depot_idx].pos = (x, y)
            x_y.append([x, y])
            demands.append(0)

    demands = np.array(demands)
    x_y = np.array(x_y)
    demands = np.concatenate((demands[-num_depots:], demands[:-num_depots]))
    x_y = np.concatenate((x_y[-num_depots:], x_y[:-num_depots]))
    return x_y, demands, np.array([capcity]), num_depots


def creat_instance(path):
    citys, demand, capcity, numdepots = load_problem(path)
    nodes = citys.copy()
    n_nodes = citys.shape[0]
    ys_demand = demand.copy()
    ys_capacity = capcity.copy()
    demand = demand.reshape(-1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    citys = scaler.fit_transform(citys)

    demand_max = np.max(demand)
    demand = demand / demand_max
    capcity = capcity / demand_max

    edge_vectors = citys[:, None, :] - citys[None, :, :]
    edges = np.linalg.norm(edge_vectors, axis=-1, keepdims=True)
    edges = edges.reshape(-1, 1)

    node_ids = torch.arange(n_nodes, dtype=torch.long)
    edges_index = torch.cartesian_prod(node_ids, node_ids).t().contiguous()
    return citys, edges, demand, edges_index, capcity, nodes, n_nodes, ys_demand, ys_capacity, numdepots


def creat_data(path, num_samples=1, batch_size=1):
    datas = []
    datas1 = []

    for _ in range(num_samples):
        citys, edges, demand, edges_index, capcity, nodes, n_nodes, ys_demand, ys_capacity, numdepots = creat_instance(path)
        datas.append(
            Data(
                x=torch.from_numpy(citys).float(),
                edge_index=edges_index,
                edge_attr=torch.from_numpy(edges).float(),
                demand=torch.tensor(demand).unsqueeze(-1).float(),
                capcity=torch.tensor(capcity).unsqueeze(-1).float(),
            )
        )
        datas1.append(
            Data(
                x=torch.from_numpy(nodes).float(),
                edge_index=edges_index,
                edge_attr=torch.from_numpy(edges).float(),
                demand=torch.tensor(ys_demand).unsqueeze(-1).float(),
                capcity=torch.tensor(ys_capacity).unsqueeze(-1).float(),
            )
        )

    dl = DataLoader(datas, batch_size=batch_size)
    dl1 = DataLoader(datas1, batch_size=batch_size)
    return dl, nodes, n_nodes, ys_demand, ys_capacity, dl1, numdepots


if __name__ == "__main__":
    creat_instance("./data/p01")
