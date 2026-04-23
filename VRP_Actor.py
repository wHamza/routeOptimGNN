import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from runtime_utils import get_num_depots, resolve_rollout_steps
from vrpUpdate import update_mask, update_route_counter, update_state

INIT = False
max_grad_norm = 2
n_nodes = 21


class GatConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0):
        super(GatConv, self).__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels + edge_channels, out_channels)
        if INIT:
            for name, p in self.named_parameters():
                if "weight" in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif "bias" in name:
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


class Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3, n_heads=4):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(input_node_dim, hidden_node_dim)
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.be = nn.BatchNorm1d(hidden_edge_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)

        self.convs1 = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for _ in range(conv_layers)]
        )
        if INIT:
            for name, p in self.named_parameters():
                if "weight" in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif "bias" in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        batch_size = data.num_graphs
        x = torch.cat([data.x, data.demand], dim=-1)
        x = self.fc_node(x)
        x = self.bn(x)
        edge_attr = self.fc_edge(data.edge_attr)
        edge_attr = self.be(edge_attr)

        for conv in self.convs1:
            x = x + conv(x, data.edge_index, edge_attr)

        return x.reshape((batch_size, -1, self.hidden_node_dim))


class Attention1(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        super(Attention1, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if INIT:
            for name, p in self.named_parameters():
                if "weight" in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif "bias" in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask):
        batch_size, n_nodes, _ = context.size()
        q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        k = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        v = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        compatibility = self.norm * torch.matmul(q, k.transpose(2, 3))
        compatibility = compatibility.squeeze(2)
        attn_mask = mask.unsqueeze(1).expand_as(compatibility)
        masked = compatibility.masked_fill(attn_mask.bool(), float("-inf"))

        scores = F.softmax(masked, dim=-1).unsqueeze(2)
        output = torch.matmul(scores, v).squeeze(2).view(batch_size, self.hidden_dim)
        return self.fc(output)


class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = Attention1(n_heads, 1, input_dim, hidden_dim)
        if INIT:
            for name, p in self.named_parameters():
                if "weight" in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif "bias" in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask, temperature):
        x = self.mhalayer(state_t, context, mask)

        batch_size, n_nodes, _ = context.size()
        q = x.view(batch_size, 1, -1)
        k = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = self.norm * torch.matmul(q, k.transpose(1, 2)).squeeze(1)
        logits = torch.tanh(compatibility) * 10
        logits = logits.masked_fill(mask.bool(), float("-inf"))
        return F.softmax(logits / temperature, dim=-1)


class Decoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.prob = ProbAttention(8, input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if INIT:
            for name, p in self.named_parameters():
                if "weight" in name and len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif "bias" in name:
                    nn.init.constant_(p, 0)

    def forward(self, encoder_inputs, pool, capcity, demand, n_steps, num_depots, temperature, greedy=False, max_routes=None):
        mask1 = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))

        dynamic_capcity = capcity.view(encoder_inputs.size(0), -1)
        demands = demand.view(encoder_inputs.size(0), encoder_inputs.size(1))
        index = torch.zeros(encoder_inputs.size(0), device=encoder_inputs.device, dtype=torch.long)
        route_count = torch.ones(encoder_inputs.size(0), device=encoder_inputs.device, dtype=torch.long)

        log_ps = []
        actions = []
        pooled_context = self.fc1(pool)

        for step_idx in range(n_steps):
            if not mask1[:, num_depots:].eq(0).any():
                break

            if step_idx == 0:
                decoder_state = encoder_inputs[:, 0, :]

            decoder_input = torch.cat([decoder_state, dynamic_capcity], dim=-1)
            decoder_input = self.fc(decoder_input) + pooled_context

            if step_idx == 0:
                mask, mask1 = update_mask(
                    demands,
                    dynamic_capcity,
                    index.unsqueeze(-1),
                    mask1,
                    step_idx,
                    num_depots,
                    route_count=route_count,
                    max_routes=max_routes,
                )

            probs = self.prob(decoder_input, encoder_inputs, mask, temperature)
            dist = Categorical(probs)

            if greedy:
                _, index = probs.max(dim=-1)
            else:
                index = dist.sample()

            actions.append(index.unsqueeze(1))
            log_p = dist.log_prob(index)
            is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
            log_ps.append((log_p * (1.0 - is_done)).unsqueeze(1))

            dynamic_capcity = update_state(
                demands,
                dynamic_capcity,
                index.unsqueeze(-1),
                num_depots,
                float(capcity.view(-1)[0].item()),
            )
            route_count = update_route_counter(index.unsqueeze(-1), route_count, num_depots, mask1)
            mask, mask1 = update_mask(
                demands,
                dynamic_capcity,
                index.unsqueeze(-1),
                mask1,
                step_idx,
                num_depots,
                route_count=route_count,
                max_routes=max_routes,
            )

            decoder_state = torch.gather(
                encoder_inputs,
                1,
                index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1, encoder_inputs.size(2)),
            ).squeeze(1)

        log_ps = torch.cat(log_ps, dim=1)
        actions = torch.cat(actions, dim=1)
        return actions, log_ps.sum(dim=1)


class Model(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
        super(Model, self).__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.decoder = Decoder1(hidden_node_dim, hidden_node_dim)

    def forward(self, datas, n_steps=None, num_depots=None, greedy=False, T=1, max_routes=None):
        if isinstance(num_depots, bool):
            if not isinstance(greedy, bool):
                T = greedy
            greedy = num_depots
            num_depots = None

        n_steps = resolve_rollout_steps(datas, n_steps)
        if num_depots is None:
            num_depots = get_num_depots(datas.x.size(0) // datas.num_graphs)

        x = self.encoder(datas)
        pooled = x.mean(dim=1)
        actions, log_p = self.decoder(
            x,
            pooled,
            datas.capcity,
            datas.demand,
            n_steps,
            int(num_depots),
            T,
            bool(greedy),
            max_routes=max_routes,
        )
        return actions, log_p
