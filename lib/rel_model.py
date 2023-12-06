# --------------------------------------------------------
# RP-FEM: Relational Prior Knowledge Graphs for Detection and Instance Segmentation
# Copyright (c) 2023
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
from .rel_graph import RelationGraph
from .context_update import ContextUpdate

class RelModel(nn.Module):
    def __init__(self, cfg=None, num_gt_heads=2, num_gt_rel_heads=2, num_gt_layers=2,
                 rel_graph_embed_dim=256, rel_graph_out_dim=1024,
                 edge_in=1024, edge_hidden=1024, edge_out=1024,
                 node_in=1024, node_hidden=1024, node_out=1024):
        super(RelModel, self).__init__()

        self.rel_info = cfg.REL_NET.PRIORS_ACTIVE
        self.A_size = (cfg.MODEL.RPN.POST_NMS_TOPK_TEST, cfg.MODEL.RPN.POST_NMS_TOPK_TEST, cfg.REL_NET.OUT_NODE_DIM)
        self.batch_per_gpu = int(cfg.SOLVER.IMS_PER_BATCH / cfg.SOLVER.GPUS)
        self.edge_out = edge_out

        if self.rel_info:
            self.relation_module = RelationGraph(cfg=cfg,
                                                 batch_per_gpu=self.batch_per_gpu,
                                                 node_in=node_in,
                                                 rel_graph_embed_dim=rel_graph_embed_dim,
                                                 rel_graph_out_dim=rel_graph_out_dim,
                                                 num_heads=num_gt_rel_heads,
                                                 relation_types=cfg.REL_NET.RELATION)
            self.n_rels = self.relation_module.n_rels

        self.cu = ContextUpdate(num_heads=num_gt_heads,
                                num_gt_layers=num_gt_layers,
                                in_edge_dim=edge_in,
                                hidden_edge_dim=edge_hidden,
                                out_edge_dim=edge_out,
                                in_node_dim=node_in,
                                hidden_node_dim=node_hidden,
                                out_node_dim=node_out)

    def forward(self, box_features):

        if not self.training:
            self.batch_per_gpu = 1

        box_features = box_features.view(self.batch_per_gpu, -1, box_features.shape[-1])
        box_features_batch = box_features.clone()

        # Get the prior informed adjacency matrix
        if self.rel_info:

            # A_matrix has shape (n_nodes x n_nodes x latent_dim)
            A_matrix = self.relation_module(box_features_batch)
            A_matrix = A_matrix.view(self.batch_per_gpu, box_features_batch.shape[1] * box_features_batch.shape[1], -1)

        # If no prior is used, use fully connected adjacency matrix
        else:
            A_matrix = torch.rand((box_features_batch.shape[0], box_features_batch.shape[0], self.edge_out), requires_grad=True)

        # Do context update with fully connected or prior informed adjacency matrix
        box_features_updated = self.cu(init_node_emb=box_features_batch, init_edge_emb=A_matrix)

        return box_features_updated