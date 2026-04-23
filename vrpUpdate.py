import torch


def update_state(demand, dynamic_capcity, selected, num_depot, c=20):
    current_demand = torch.gather(demand, 1, selected)
    dynamic_capcity = dynamic_capcity - current_demand

    selected_nodes = selected.squeeze(-1)
    for depot_idx in range(num_depot):
        depot_mask = selected_nodes.eq(depot_idx)
        if depot_mask.any():
            dynamic_capcity[depot_mask] = c

    return dynamic_capcity.detach()


def update_route_counter(selected, route_count, num_depot, visited_mask):
    selected_nodes = selected.squeeze(-1)
    returned_to_depot = selected_nodes.lt(num_depot)
    has_visited_customer = visited_mask[:, num_depot:].bool().any(dim=1)
    completed_route = returned_to_depot & has_visited_customer
    return route_count + completed_route.to(route_count.dtype)


def update_mask(demand, capcity, selected, mask, i, num_depot, route_count=None, max_routes=None):
    selected_nodes = selected.squeeze(-1)
    go_depot = torch.zeros(selected_nodes.size(0), dtype=torch.bool, device=selected.device)
    for depot_idx in range(num_depot):
        go_depot |= selected_nodes.eq(depot_idx)

    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)

    if (~go_depot).any():
        mask1[~go_depot, :num_depot] = 0

    if go_depot.any():
        mask1[go_depot, :num_depot] = 1

    if i + num_depot > demand.size(1):
        is_done = mask1[:, num_depot:].sum(1) >= (demand.size(1) - num_depot)
        if is_done.any():
            mask1[is_done, 0] = 0

    if route_count is not None and max_routes is not None:
        route_limit_reached = route_count >= max_routes
        if route_limit_reached.any():
            has_unserved_customers = mask1[:, num_depot:].eq(0).any(dim=1)
            lock_depot = route_limit_reached & has_unserved_customers
            if lock_depot.any():
                mask1[lock_depot, :num_depot] = 1

    mask = (demand > capcity) | mask1.bool()
    return mask.detach(), mask1.detach()
