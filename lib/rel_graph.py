# --------------------------------------------------------
# RP-FEM: Relational Prior Knowledge Graphs for Detection and Instance Segmentation
# Copyright (c) 2023
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import json
import torch
import torch.nn as nn
from mhna import MultiHeadedNeighborhoodAttention

COCO_final_layer = "/path/to/COCO_box_predictor.pt"
ROOT = "path/to/root/"
GRAPHS_SUBDIR = "path/to/RPKGs/"
REL_TYPES = ["co-occurence", "spatial-relations", "relative-distance"]

class RelationGraph(nn.Module):
    def __init__(self, cfg, batch_per_gpu, node_in, rel_graph_embed_dim, rel_graph_out_dim, num_heads,
                 relation_types=REL_TYPES,
                 class_weights_path=COCO_final_layer):
        super().__init__()

        # Load relational metrics
        if not isinstance(relation_types, list):
            self.relation_types = [relation_types]
        else:
            self.relation_types = relation_types
        if len(self.relation_types) > 1:
            self.graphs = torch.cat([self.read_relation_graph_values(cfg, relation_type) for relation_type in self.relation_types], dim=2)
        else:
            self.graphs = self.read_relation_graph_values(cfg, self.relation_types[0])
        self.keys, self.label_idx_mapping, self.idx_label_mapping = self.read_relation_graph_classes(cfg, self.relation_types[0])

        # Retrieve class weights (2024 x 1601) in the correct mapping order
        # For this we use the last layer of a detection network on VG or COCO
        # Ours can be downloaded here: https://surfdrive.surf.nl/files/index.php/s/P0nJ6PNTN4pIyd6/download
        class_weights_cpu = torch.load(class_weights_path, map_location='cpu')['weight'][1:]
        self.class_weights = torch.clone(class_weights_cpu)

        self.n_rels = self.graphs.shape[-1]
        self.prior_enhancer = MultiHeadedNeighborhoodAttention(prior_graphs=self.graphs,
                                                               prior_class_weights=self.class_weights,
                                                               batch_per_gpu=batch_per_gpu,
                                                               in_dim=node_in,
                                                               emb_dim=rel_graph_embed_dim,
                                                               out_dim=rel_graph_out_dim,
                                                               n_heads=num_heads)

        self.n_nodes = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE

        self.relation_to_file = {
            "co-occurence": ROOT + GRAPHS_SUBDIR + "co-occurence.json",
            "spatial-relations": ROOT + GRAPHS_SUBDIR + "relative-orientation.json",
            "relative-distance": ROOT + GRAPHS_SUBDIR + "relative-distance.json"}

    def read_relation_graph_classes(self, relation_type):
        assert relation_type in list(self.relation_to_file.keys())
        with open(self.relation_to_file[relation_type]) as json_file:
            data = json.load(json_file)
        keys = list(data.keys())
        label_idx_mapping, idx_label_mapping = {}, {}
        for index, key in enumerate(keys):
            label_idx_mapping[key] = index
            idx_label_mapping[index] = key
        return keys, label_idx_mapping, idx_label_mapping

    def read_relation_graph_values(self, relation_type):
        assert relation_type in list(self.relation_to_file.keys())
        with open(self.relation_to_file[relation_type]) as json_file:
            data = json.load(json_file)
        keys = list(data.keys())
        label_idx_mapping, idx_label_mapping = {}, {}
        for index, key in enumerate(keys):
            label_idx_mapping[key] = index
            idx_label_mapping[index] = key

        if isinstance(data[key][key], float):
            values = torch.zeros((len(data), len(data)))
        else:
            values = torch.zeros((len(data), len(data), len(data[key][key])))
        for c1, c_c in enumerate(list(data.values())):
            for c2, c_r in enumerate(list(c_c.values())):
                values[c1][c2] = torch.tensor(c_r)
        if isinstance(c_r, float):
            return values.unsqueeze(2)
        return values

    def label_to_index(self, label):
        return self.label_idx_mapping[label]

    def idx_label_mapping(self, idx):
        return self.idx_label_mapping[idx]

    def forward(self, x):

        # Shape: relation_types x (batch_size x n_nodes x embed_dim)
        new_A_matrix = self.prior_enhancer(x)

        return new_A_matrix