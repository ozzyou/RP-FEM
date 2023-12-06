# --------------------------------------------------------
# RP-FEM: Relational Prior Knowledge Graphs for Detection and Instance Segmentation
# Copyright (c) 2023
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import pygmtools as pygm
pygm.BACKEND = 'pytorch'

class MultiHeadedNeighborhoodAttention(nn.Module):
    def __init__(self, prior_graphs, prior_class_weights, batch_per_gpu, in_dim, emb_dim, out_dim, n_heads):
        super().__init__()

        self.batch_per_gpu = batch_per_gpu
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        self.prior_graphs = prior_graphs
        self.prior_class_weights = prior_class_weights

        self.n_relations = prior_graphs.shape[-1]
        self.n_prior_classes = self.prior_class_weights.shape[0]

        self.q_proj = nn.Linear(in_dim, emb_dim)
        self.k_proj = nn.Linear(prior_class_weights.shape[-1], emb_dim)
        self.v_proj = nn.Linear(prior_class_weights.shape[-1], emb_dim)
        self.n_proj = nn.Linear(emb_dim, out_dim)
        self.a_proj = nn.Linear(1, 1024)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.n_proj.weight)
        nn.init.xavier_uniform_(self.a_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.n_proj.bias.data.fill_(0)
        self.a_proj.bias.data.fill_(0)

    def forward(self, x):

        if not self.training:
            self.batch_per_gpu = 1

        batch_size, n_nodes, _ = x.shape

        q = self.q_proj(x)
        k, v = self.prior_class_weights, self.prior_class_weights
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape(self.batch_per_gpu, n_nodes, self.n_heads, self.head_dim)
        k = k.reshape(self.n_prior_classes, self.n_heads, self.head_dim)
        v = v.reshape(self.n_prior_classes, self.n_heads, self.head_dim)

        # Bring heads to first dimension
        q = q.permute(0, 2, 1, 3)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        values = F.scaled_dot_product_attention(query=q, key=k, value=v)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(self.batch_per_gpu, n_nodes, self.n_heads * self.head_dim)

        # Once we have our subgraph values, we try to find the subgraph in the big graph
        nodes_SG = self.n_proj(values)
        nodes_PKG = self.prior_class_weights.unsqueeze(0).repeat(batch_size, 1, 1)

        A_SG = torch.ones(batch_size, nodes_SG.shape[1], nodes_SG.shape[1])
        A_PKG = self.prior_graphs.squeeze().repeat(batch_size, 1, 1)

        n_SG = torch.tensor([n_nodes] * batch_size)
        n_PKG = torch.tensor([nodes_PKG.shape[1]] * batch_size)

        nodes_SG = F.normalize(nodes_SG)
        nodes_PKG = F.normalize(nodes_PKG)

        X, net = pygm.pca_gm(nodes_SG, nodes_PKG, A_SG, A_PKG, n_SG, n_PKG, return_network=True)
        X_t = torch.transpose(X, 1, 2)

        batched_RPKG = self.prior_graphs.squeeze().repeat(batch_size, 1, 1)
        batched_RPKG_reduced = torch.matmul(batched_RPKG, X_t)
        batched_RPKG_reduced_t = torch.transpose(batched_RPKG_reduced, 1, 2)

        A = torch.matmul(batched_RPKG_reduced_t, X_t).unsqueeze(-1)
        edge_emb = self.a_proj(A)

        return edge_emb
