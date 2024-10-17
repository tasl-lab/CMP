import time

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtr.models_v2v4real.model import MotionTransformer
from mtr.utils import common_utils, loss_utils, motion_utils
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, activator='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim
        self.num_layers = num_layers
        self.activator = activator

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == self.num_layers - 1:
                self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            else:
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i != self.num_layers - 1:
                if self.activator == 'relu':
                    x = F.relu(x)
                elif self.activator == 'mish':
                    x = F.mish(x)
                else:
                    raise NotImplementedError
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, kdim=128, vdim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, kdim=kdim, vdim=vdim)

    def forward(self, query, key, value, key_padding_mask=None):
        return self.attn(query, key, value, key_padding_mask=key_padding_mask)[0]  # Get the output of (outputs, weights)

class MotionAggregatorMLP(nn.Module):
    def __init__(self):
        super(MotionAggregatorMLP, self).__init__()
        self.type = 'MLP'

        # Trajectory Encoder: A simple linear layer to process trajectories
        self.trajectory_encoder = nn.Linear(5, 128)

        # BEV Feature Encoder: A simple CNN to process BEV features
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )

        # Fusion Layer: Combining trajectory and BEV features
        self.fusion_layer = nn.Linear(256, 128)

        # Decoder: Generating the final output for trajectories
        self.decoder = nn.Linear(128, 5)

        # Confidence Score Estimator: Estimating new confidence scores
        self.confidence_estimator = nn.Linear(128, 1)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        # Merge all predictions in a giant tensor.
        trajectories = torch.cat(features_to_aggregate, dim=0)  # (num_center_objects, num_modes, num_timestamps, 5)
        pred_scores = torch.cat(pred_scores, dim=0)  # (num_center_objects, num_modes)

        # Sync device.
        bev_features = bev_features.to(trajectories.device)

        # Flatten trajectories for processing: shape (N * M * T, 5)
        num_tracks, num_modes, num_timestamps, _ = trajectories.shape
        trajectories = trajectories.view(-1, 5)

        # Process trajectories
        traj_features = self.trajectory_encoder(trajectories)  # (N * M * T, 128)

        # Process BEV features
        bev_features = self.bev_encoder(bev_features)  # (1, 128)
        bev_features = bev_features.repeat(num_tracks * num_modes * num_timestamps, 1)  # (N * M * T, 128)

        # Fuse features
        fused_features = torch.cat((traj_features, bev_features), dim=1)  # (N * M * T, 256)
        fused_features = self.fusion_layer(fused_features)  # (N * M * T, 128)

        # Decode to get final output for trajectories
        output_trajectories = self.decoder(fused_features)  # (N * M * T, 7)
        output_trajectories = output_trajectories.view(num_tracks, num_modes, num_timestamps, 5)

        # Estimate new confidence scores
        fused_features_for_scores = fused_features.view(num_tracks, num_modes, num_timestamps, -1).mean(dim=2)  # (N, M, 128)
        new_confidence_scores = self.confidence_estimator(fused_features_for_scores)  # (N, M, 1)
        new_confidence_scores = new_confidence_scores.squeeze(-1)  # (N, M)
        new_confidence_scores = F.softmax(new_confidence_scores, dim=-1)  # Apply softmax to get probabilities

        # Select only the trajectories and scores for batch_sample_count objects
        output_trajectories = output_trajectories[:batch_sample_count]
        new_confidence_scores = new_confidence_scores[:batch_sample_count]

        return output_trajectories, new_confidence_scores

class MotionAggregatorMLPV2(nn.Module):
    def __init__(self):
        super(MotionAggregatorMLPV2, self).__init__()
        self.type = 'MLPV2'

        # Trajectory Encoder: A simple linear layer to process trajectories
        self.trajectory_encoder = MLP(5, 128)

        # BEV Feature Encoder: A simple CNN to process BEV features
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )

        # Fusion Layer: Combining trajectory and BEV features
        self.fusion_layer = MLP(256, 128)

        # Decoder: Generating the final output for trajectories
        self.decoder = MLP(128, 5)

        # Confidence Score Estimator: Estimating new confidence scores
        self.confidence_estimator = MLP(128, 1)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        # Merge all predictions in a giant tensor.
        trajectories = torch.cat(features_to_aggregate, dim=0)  # (num_center_objects, num_modes, num_timestamps, 5)
        pred_scores = torch.cat(pred_scores, dim=0)  # (num_center_objects, num_modes)

        # Sync device.
        bev_features = bev_features.to(trajectories.device)

        # Flatten trajectories for processing: shape (N * M * T, 5)
        num_tracks, num_modes, num_timestamps, _ = trajectories.shape
        trajectories = trajectories.view(-1, 5)

        # Process trajectories
        traj_features = self.trajectory_encoder(trajectories)  # (N * M * T, 128)

        # Process BEV features
        bev_features = self.bev_encoder(bev_features)  # (1, 128)
        bev_features = bev_features.repeat(num_tracks * num_modes * num_timestamps, 1)  # (N * M * T, 128)

        # Fuse features
        fused_features = torch.cat((traj_features, bev_features), dim=1)  # (N * M * T, 256)
        fused_features = self.fusion_layer(fused_features)  # (N * M * T, 128)

        # Decode to get final output for trajectories
        output_trajectories = self.decoder(fused_features)  # (N * M * T, 7)
        output_trajectories = output_trajectories.view(num_tracks, num_modes, num_timestamps, 5)

        # Estimate new confidence scores
        fused_features_for_scores = fused_features.view(num_tracks, num_modes, num_timestamps, -1).mean(dim=2)  # (N, M, 128)
        new_confidence_scores = self.confidence_estimator(fused_features_for_scores)  # (N, M, 1)
        new_confidence_scores = new_confidence_scores.squeeze(-1)  # (N, M)
        new_confidence_scores = F.softmax(new_confidence_scores, dim=-1)  # Apply softmax to get probabilities

        # Select only the trajectories and scores for batch_sample_count objects
        output_trajectories = output_trajectories[:batch_sample_count]
        new_confidence_scores = new_confidence_scores[:batch_sample_count]

        return output_trajectories, new_confidence_scores

class MotionAggregatorGCN(nn.Module):
    def __init__(self):
        super(MotionAggregatorGCN, self).__init__()
        self.type = 'GCN'

        # Graph Convolution layer
        self.gcn_conv_global = GCNConv(512, 512)

        # Feature processing layers
        self.feature_encoder = MLP(50 * 5, 512)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )
        self.map_encoder = MLP(49 * 256, 1024)

        # Trajectory decoder to decode a single trajectory embedding into a standard trajectory.
        self.trajectory_decoder = MLP(6 * 512 + 128 + 1024, 6 * 50 * 5)

        # Confidence Score Estimator: Estimating new confidence scores among modes given the BEV feature.
        self.confidence_estimator = MLP(6 * 512 + 128 + 1024, 6)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """

        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        map_features = (map_polylines_feature[0] * map_valid_mask[0].unsqueeze(-1)).view(ego_cav_traj_count, -1) # (ego_cav_traj_count, 49 * 256)

        # Reshape so that each node is a single trajectory.
        global_nodes = []
        edge_indices = []
        for cav_feature, pred_score in zip(features_to_aggregate, pred_scores):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Weight incoming feature by its confidence score.
            cav_feature = cav_feature * pred_score.unsqueeze(-1).unsqueeze(-1)

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav, num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav, num_modes, 50*5 -> num_center_objects_this_cav, num_modes, 512

            # Split into individual agents.
            agents = [agent for agent in torch.unbind(cav_feature_embedding, dim=0)]

            # Within each agent, split into mode, with each mode being a node.
            for agent in agents:
                modes = [mode for mode in torch.unbind(agent, dim=0)]

                # We only add edges to nodes that are already in the graph.
                # i.e. we exclude connections within the modes. Otherwise,
                # hyper-smoothing will lead to mode collapse.
                curr_node_count = len(global_nodes)
                for i in range(len(modes)):
                    node_id = curr_node_count + i
                    for j in range(curr_node_count):
                        edge_indices.append([node_id, j])
                        edge_indices.append([j, node_id])

                global_nodes.extend(modes)

        # Create global edge list.
        edge_indices = torch.tensor(edge_indices, dtype=torch.long, device=device).t().contiguous()

        # Global graph message passing.
        global_nodes = torch.stack(global_nodes, dim=0)  # (num_cavs * num_center_objects_per_cav * num_modes, 512)
        if edge_indices.shape[0] != 0:
            global_nodes = self.gcn_conv_global(global_nodes, edge_indices)

        # Extract ego nodes at index 0.
        ego_node_feature = global_nodes[0:ego_cav_traj_count * num_modes]
        ego_node_feature = ego_node_feature.view(ego_cav_traj_count, -1)  # (ego_cav_traj_count, num_modes * 512)

        # Fuse BEV.
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)
        bev_feature_embedding = bev_feature_embedding.repeat(ego_cav_traj_count, 1)  # (ego_cav_traj_count, 128)
        map_feature_embedding = self.map_encoder(map_features)  # (num_predictions_for_this_agent, 1024)
        cav_feature_embedding = torch.cat([ego_node_feature, bev_feature_embedding, map_feature_embedding],
                                          dim=-1)  # (num_center_objects_this_cav * num_modes, 512 + 128 + 1024)

        # Decode the features to get trajectories
        output_trajectories = self.trajectory_decoder(cav_feature_embedding)
        output_trajectories = output_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)

        # Re-estimate confidence scores
        new_confidence_scores = self.confidence_estimator(cav_feature_embedding)  # Apply softmax to get probabilities
        new_confidence_scores = new_confidence_scores.view(ego_cav_traj_count, num_modes)
        new_confidence_scores = nn.functional.softmax(new_confidence_scores, dim=-1)

        return output_trajectories, new_confidence_scores

class MotionAggregatorMOE(nn.Module):
    """
    Original MOE from Hengbo.
    """
    def __init__(self):
        super(MotionAggregatorMOE, self).__init__()
        self.type = 'MOE'

        # Feature processing layers
        self.feature_encoder = MLP(50 * 5, 512)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )
        self.map_encoder = MLP(49 * 256, 1024)

        # Trajectory decoder to decode a single trajectory embedding into a standard trajectory.
        self.trajectory_decoder = MLP(512, 50 * 5)

        # Estimator that for each mode and a BEV feature, estimate its gating factor
        self.gating_factor_estimators = nn.ModuleList([MLP(512 + 128 + 1024, 1) for _ in range(6)])

        # Score Estimator.
        self.score_estimator = MLP(6 * 512, 6)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores, center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav * num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav*num_modes, 50*5 -> num_center_objects_this_cav*num_modes, 512

            # Separate by agent.
            agent_feature_embeddings = [agent_feature_embedding for agent_feature_embedding in torch.unbind(cav_feature_embedding, dim=0)]
            center_objects_id = torch.from_numpy(center_objects_id).unsqueeze(-1).repeat(1, num_modes).flatten()  # num_center_objects_per_cav -> num_center_objects_per_cav*num_modes
            center_objects_id = center_objects_id.tolist()
            assert len(agent_feature_embeddings) == len(center_objects_id)

            for agent_feature_embedding, center_object_id in zip(agent_feature_embeddings, center_objects_id):
                predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [agent_feature_embedding]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        ego_map_features = map_polylines_feature[0]
        ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):
            agent_feature_embeddings = predictions_by_cav[center_object_id]  # list of (512)
            num_predictions_for_this_agent_with_modes = len(agent_feature_embeddings)

            # Stack predictions from different agents.
            agent_feature_embeddings = torch.stack(agent_feature_embeddings, dim=0)  # (num_predictions_for_this_agent*num_modes, 512)

            # Fuse BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 128)
            map_feature_embedding_this_agent = self.map_encoder((ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(num_predictions_for_this_agent_with_modes, 1) # (num_predictions_for_this_agent*num_modes, 1024)
            cav_feature_embedding = torch.cat([agent_feature_embeddings,
                                               bev_feature_embedding_this_agent,
                                               map_feature_embedding_this_agent],
                                              dim=-1)  # (num_predictions_for_this_agent*num_modes, 512 + 128 + 1024)

            # Estimate gating factor for each mode and aggregate.
            assert len(self.gating_factor_estimators) == num_modes
            agent_prediction_per_mode = []
            for mode_idx in range(num_modes):
                gating_factor = self.gating_factor_estimators[mode_idx](cav_feature_embedding)  # (num_predictions_for_this_agent*num_modes, 1)
                gating_factor = nn.functional.softmax(gating_factor, dim=0)

                # Combine the predictions.
                weights_agent_feature_embeddings = gating_factor * agent_feature_embeddings  # (num_predictions_for_this_agent*num_modes, 512)
                aggregated_agent_feature_embedding = weights_agent_feature_embeddings.sum(dim=0)  # (512)
                agent_prediction_per_mode.append(aggregated_agent_feature_embedding)

            # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            combined_prediction = torch.stack(agent_prediction_per_mode, dim=0)  # (num_modes, 512)
            final_prediction_embeddings.append(combined_prediction)

            # Estimate the scores.
            scores = self.score_estimator(combined_prediction.view(-1))  # (num_modes)
            final_scores.append(scores)

        final_prediction_embeddings = torch.stack(final_prediction_embeddings, dim=0)  # (num_center_objects_first_cav, num_modes, 512)
        final_trajectories = self.trajectory_decoder(final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, 50 * 5)
        final_trajectories = final_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)
        final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, final_scores

class MotionAggregatorMOEV2(nn.Module):
    """
    Compared to the orginal version, here we try making the map embedding from 1024->128.
    """

    def __init__(self):
        super(MotionAggregatorMOEV2, self).__init__()
        self.type = 'MOEV2'

        # Feature processing layers
        self.feature_encoder = MLP(50 * 5, 512)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )
        self.map_encoder = MLP(49 * 256, 128)

        # Trajectory decoder to decode a single trajectory embedding into a standard trajectory.
        self.trajectory_decoder = MLP(512, 50 * 5)

        # Estimator that for each mode and a BEV feature, estimate its gating factor
        self.gating_factor_estimators = nn.ModuleList([MLP(512 + 128 + 128, 1) for _ in range(6)])

        # Score Estimator.
        self.score_estimator = MLP(6 * 512, 6)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores, center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav * num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav*num_modes, 50*5 -> num_center_objects_this_cav*num_modes, 512

            # Separate by agent.
            agent_feature_embeddings = [agent_feature_embedding for agent_feature_embedding in torch.unbind(cav_feature_embedding, dim=0)]
            center_objects_id = torch.from_numpy(center_objects_id).unsqueeze(-1).repeat(1, num_modes).flatten()  # num_center_objects_per_cav -> num_center_objects_per_cav*num_modes
            center_objects_id = center_objects_id.tolist()
            assert len(agent_feature_embeddings) == len(center_objects_id)

            for agent_feature_embedding, center_object_id in zip(agent_feature_embeddings, center_objects_id):
                predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [agent_feature_embedding]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        ego_map_features = map_polylines_feature[0]
        ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):
            agent_feature_embeddings = predictions_by_cav[center_object_id]  # list of (512)
            num_predictions_for_this_agent_with_modes = len(agent_feature_embeddings)

            # Stack predictions from different agents.
            agent_feature_embeddings = torch.stack(agent_feature_embeddings, dim=0)  # (num_predictions_for_this_agent*num_modes, 512)

            # Fuse BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 128)
            map_feature_embedding_this_agent = self.map_encoder((ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(num_predictions_for_this_agent_with_modes, 1) # (num_predictions_for_this_agent*num_modes, 128)
            cav_feature_embedding = torch.cat([agent_feature_embeddings,
                                               bev_feature_embedding_this_agent,
                                               map_feature_embedding_this_agent],
                                              dim=-1)  # (num_predictions_for_this_agent*num_modes, 512 + 128 + 128)

            # Estimate gating factor for each mode and aggregate.
            assert len(self.gating_factor_estimators) == num_modes
            agent_prediction_per_mode = []
            for mode_idx in range(num_modes):
                gating_factor = self.gating_factor_estimators[mode_idx](cav_feature_embedding)  # (num_predictions_for_this_agent*num_modes, 1)
                gating_factor = nn.functional.softmax(gating_factor, dim=0)

                # Combine the predictions.
                weights_agent_feature_embeddings = gating_factor * agent_feature_embeddings  # (num_predictions_for_this_agent*num_modes, 512)
                aggregated_agent_feature_embedding = weights_agent_feature_embeddings.sum(dim=0)  # (512)
                agent_prediction_per_mode.append(aggregated_agent_feature_embedding)

            # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            combined_prediction = torch.stack(agent_prediction_per_mode, dim=0)  # (num_modes, 512)
            final_prediction_embeddings.append(combined_prediction)

            # Estimate the scores.
            scores = self.score_estimator(combined_prediction.view(-1))  # (num_modes)
            final_scores.append(scores)

        final_prediction_embeddings = torch.stack(final_prediction_embeddings, dim=0)  # (num_center_objects_first_cav, num_modes, 512)
        final_trajectories = self.trajectory_decoder(final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, 50 * 5)
        final_trajectories = final_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)
        final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, final_scores

class MotionAggregatorMOEV3(nn.Module):
    """
    Compared to the orginal version, here we try making the traj embedding 512 -> 256, and BEV/Map 128 -> 64.
    """

    def __init__(self):
        super(MotionAggregatorMOEV3, self).__init__()
        self.type = 'MOEV3'

        # Feature processing layers
        self.feature_encoder = MLP(50 * 5, 256)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 64)
        )
        self.map_encoder = MLP(49 * 256, 64)

        # Trajectory decoder to decode a single trajectory embedding into a standard trajectory.
        self.trajectory_decoder = MLP(256, 50 * 5)

        # Estimator that for each mode and a BEV feature, estimate its gating factor
        self.gating_factor_estimators = nn.ModuleList([MLP(256 + 64 + 64, 1) for _ in range(6)])

        # Score Estimator.
        self.score_estimator = MLP(6 * 256, 6)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 64)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores, center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav * num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav*num_modes, 50*5 -> num_center_objects_this_cav*num_modes, 256

            # Separate by agent.
            agent_feature_embeddings = [agent_feature_embedding for agent_feature_embedding in torch.unbind(cav_feature_embedding, dim=0)]
            center_objects_id = torch.from_numpy(center_objects_id).unsqueeze(-1).repeat(1, num_modes).flatten()  # num_center_objects_per_cav -> num_center_objects_per_cav*num_modes
            center_objects_id = center_objects_id.tolist()
            assert len(agent_feature_embeddings) == len(center_objects_id)

            for agent_feature_embedding, center_object_id in zip(agent_feature_embeddings, center_objects_id):
                predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [agent_feature_embedding]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        ego_map_features = map_polylines_feature[0]
        ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):
            agent_feature_embeddings = predictions_by_cav[center_object_id]  # list of (256)
            num_predictions_for_this_agent_with_modes = len(agent_feature_embeddings)

            # Stack predictions from different agents.
            agent_feature_embeddings = torch.stack(agent_feature_embeddings, dim=0)  # (num_predictions_for_this_agent*num_modes, 256)

            # Fuse BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 64)
            map_feature_embedding_this_agent = self.map_encoder((ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(num_predictions_for_this_agent_with_modes, 1) # (num_predictions_for_this_agent*num_modes, 64)
            cav_feature_embedding = torch.cat([agent_feature_embeddings,
                                               bev_feature_embedding_this_agent,
                                               map_feature_embedding_this_agent],
                                              dim=-1)  # (num_predictions_for_this_agent*num_modes, 256 + 64 + 64)

            # Estimate gating factor for each mode and aggregate.
            assert len(self.gating_factor_estimators) == num_modes
            agent_prediction_per_mode = []
            for mode_idx in range(num_modes):
                gating_factor = self.gating_factor_estimators[mode_idx](cav_feature_embedding)  # (num_predictions_for_this_agent*num_modes, 1)
                gating_factor = nn.functional.softmax(gating_factor, dim=0)

                # Combine the predictions.
                weights_agent_feature_embeddings = gating_factor * agent_feature_embeddings  # (num_predictions_for_this_agent*num_modes, 256)
                aggregated_agent_feature_embedding = weights_agent_feature_embeddings.sum(dim=0)  # (256)
                agent_prediction_per_mode.append(aggregated_agent_feature_embedding)

            # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            combined_prediction = torch.stack(agent_prediction_per_mode, dim=0)  # (num_modes, 256)
            final_prediction_embeddings.append(combined_prediction)

            # Estimate the scores.
            scores = self.score_estimator(combined_prediction.view(-1))  # (num_modes)
            final_scores.append(scores)

        final_prediction_embeddings = torch.stack(final_prediction_embeddings, dim=0)  # (num_center_objects_first_cav, num_modes, 256)
        final_trajectories = self.trajectory_decoder(final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, 50 * 5)
        final_trajectories = final_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)
        final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, final_scores

class MotionAggregatorMOEV4(nn.Module):
    """
    Compared to the V2, here we try using mish as activation.
    """

    def __init__(self):
        super(MotionAggregatorMOEV4, self).__init__()
        self.type = 'MOEV4'

        # Feature processing layers
        self.feature_encoder = MLP(50 * 5, 512, activator='mish')
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )
        self.map_encoder = MLP(49 * 256, 128, activator='mish')

        # Trajectory decoder to decode a single trajectory embedding into a standard trajectory.
        self.trajectory_decoder = MLP(512, 50 * 5, activator='mish')

        # Estimator that for each mode and a BEV feature, estimate its gating factor
        self.gating_factor_estimators = nn.ModuleList([MLP(512 + 128 + 128, 1, activator='mish') for _ in range(6)])

        # Score Estimator.
        self.score_estimator = MLP(6 * 512, 6, activator='mish')

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores, center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav * num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav*num_modes, 50*5 -> num_center_objects_this_cav*num_modes, 512

            # Separate by agent.
            agent_feature_embeddings = [agent_feature_embedding for agent_feature_embedding in torch.unbind(cav_feature_embedding, dim=0)]
            center_objects_id = torch.from_numpy(center_objects_id).unsqueeze(-1).repeat(1, num_modes).flatten()  # num_center_objects_per_cav -> num_center_objects_per_cav*num_modes
            center_objects_id = center_objects_id.tolist()
            assert len(agent_feature_embeddings) == len(center_objects_id)

            for agent_feature_embedding, center_object_id in zip(agent_feature_embeddings, center_objects_id):
                predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [agent_feature_embedding]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        ego_map_features = map_polylines_feature[0]
        ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):
            agent_feature_embeddings = predictions_by_cav[center_object_id]  # list of (512)
            num_predictions_for_this_agent_with_modes = len(agent_feature_embeddings)

            # Stack predictions from different agents.
            agent_feature_embeddings = torch.stack(agent_feature_embeddings, dim=0)  # (num_predictions_for_this_agent*num_modes, 512)

            # Fuse BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 128)
            map_feature_embedding_this_agent = self.map_encoder((ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(num_predictions_for_this_agent_with_modes, 1) # (num_predictions_for_this_agent*num_modes, 128)
            cav_feature_embedding = torch.cat([agent_feature_embeddings,
                                               bev_feature_embedding_this_agent,
                                               map_feature_embedding_this_agent],
                                              dim=-1)  # (num_predictions_for_this_agent*num_modes, 512 + 128 + 128)

            # Estimate gating factor for each mode and aggregate.
            assert len(self.gating_factor_estimators) == num_modes
            agent_prediction_per_mode = []
            for mode_idx in range(num_modes):
                gating_factor = self.gating_factor_estimators[mode_idx](cav_feature_embedding)  # (num_predictions_for_this_agent*num_modes, 1)
                gating_factor = nn.functional.softmax(gating_factor, dim=0)

                # Combine the predictions.
                weights_agent_feature_embeddings = gating_factor * agent_feature_embeddings  # (num_predictions_for_this_agent*num_modes, 512)
                aggregated_agent_feature_embedding = weights_agent_feature_embeddings.sum(dim=0)  # (512)
                agent_prediction_per_mode.append(aggregated_agent_feature_embedding)

            # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            combined_prediction = torch.stack(agent_prediction_per_mode, dim=0)  # (num_modes, 512)
            final_prediction_embeddings.append(combined_prediction)

            # Estimate the scores.
            scores = self.score_estimator(combined_prediction.view(-1))  # (num_modes)
            final_scores.append(scores)

        final_prediction_embeddings = torch.stack(final_prediction_embeddings, dim=0)  # (num_center_objects_first_cav, num_modes, 512)
        final_trajectories = self.trajectory_decoder(final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, 50 * 5)
        final_trajectories = final_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)
        final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, final_scores

class MotionAggregatorMOEV5(nn.Module):
    """
    Compared to the V2, here we try using BN at the concat features.
    """

    def __init__(self):
        super(MotionAggregatorMOEV5, self).__init__()
        self.type = 'MOEV5'

        # Feature processing layers
        self.feature_encoder = MLP(50 * 5, 512, activator='mish')
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )
        self.map_encoder = MLP(49 * 256, 128, activator='mish')

        # Trajectory decoder to decode a single trajectory embedding into a standard trajectory.
        self.trajectory_decoder = MLP(512, 50 * 5, activator='mish')

        # # LN functors.
        # self.agt_feature_bn = nn.LayerNorm(512)
        # self.bev_feature_bn = nn.LayerNorm(128)
        # self.map_feature_bn = nn.LayerNorm(128)

        # BN functors.
        self.agt_feature_bn = nn.BatchNorm1d(512)
        self.bev_feature_bn = nn.BatchNorm1d(128)
        self.map_feature_bn = nn.BatchNorm1d(128)

        # Estimator that for each mode and a BEV feature, estimate its gating factor
        self.gating_factor_estimators = nn.ModuleList([MLP(512 + 128 + 128, 1, activator='mish') for _ in range(6)])

        # Score Estimator.
        self.score_estimator = MLP(6 * 512, 6, activator='mish')

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores, center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav * num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav*num_modes, 50*5 -> num_center_objects_this_cav*num_modes, 512

            # Separate by agent.
            agent_feature_embeddings = [agent_feature_embedding for agent_feature_embedding in torch.unbind(cav_feature_embedding, dim=0)]
            center_objects_id = torch.from_numpy(center_objects_id).unsqueeze(-1).repeat(1, num_modes).flatten()  # num_center_objects_per_cav -> num_center_objects_per_cav*num_modes
            center_objects_id = center_objects_id.tolist()
            assert len(agent_feature_embeddings) == len(center_objects_id)

            for agent_feature_embedding, center_object_id in zip(agent_feature_embeddings, center_objects_id):
                predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [agent_feature_embedding]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        ego_map_features = map_polylines_feature[0]
        ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):
            agent_feature_embeddings = predictions_by_cav[center_object_id]  # list of (512)
            num_predictions_for_this_agent_with_modes = len(agent_feature_embeddings)

            # Stack predictions from different agents.
            agent_feature_embeddings = torch.stack(agent_feature_embeddings, dim=0)  # (num_predictions_for_this_agent*num_modes, 512)

            # Fuse BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 128)
            map_feature_embedding_this_agent = self.map_encoder((ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(num_predictions_for_this_agent_with_modes, 1) # (num_predictions_for_this_agent*num_modes, 128)
            
            # Batch Norm
            agent_feature_embeddings_gate = self.agt_feature_bn(agent_feature_embeddings)
            bev_feature_embedding_this_agent = self.bev_feature_bn(bev_feature_embedding_this_agent)
            map_feature_embedding_this_agent = self.map_feature_bn(map_feature_embedding_this_agent)
            
            cav_feature_embedding = torch.cat([agent_feature_embeddings_gate,
                                               bev_feature_embedding_this_agent,
                                               map_feature_embedding_this_agent],
                                              dim=-1)  # (num_predictions_for_this_agent*num_modes, 512 + 128 + 128)

            # Estimate gating factor for each mode and aggregate.
            assert len(self.gating_factor_estimators) == num_modes
            agent_prediction_per_mode = []
            for mode_idx in range(num_modes):
                gating_factor = self.gating_factor_estimators[mode_idx](cav_feature_embedding)  # (num_predictions_for_this_agent*num_modes, 1)
                gating_factor = nn.functional.softmax(gating_factor, dim=0)

                # Combine the predictions.
                weights_agent_feature_embeddings = gating_factor * agent_feature_embeddings  # (num_predictions_for_this_agent*num_modes, 512)
                aggregated_agent_feature_embedding = weights_agent_feature_embeddings.sum(dim=0)  # (512)
                agent_prediction_per_mode.append(aggregated_agent_feature_embedding)

            # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            combined_prediction = torch.stack(agent_prediction_per_mode, dim=0)  # (num_modes, 512)
            final_prediction_embeddings.append(combined_prediction)

            # Estimate the scores.
            scores = self.score_estimator(combined_prediction.view(-1))  # (num_modes)
            final_scores.append(scores)

        final_prediction_embeddings = torch.stack(final_prediction_embeddings, dim=0)  # (num_center_objects_first_cav, num_modes, 512)
        final_trajectories = self.trajectory_decoder(final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, 50 * 5)
        final_trajectories = final_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)
        final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, final_scores

class MotionAggregatorMOEV6(nn.Module):
    """
    Compared to v2, here we try feeding all features into the aggregator.
    """

    def __init__(self):
        super(MotionAggregatorMOEV6, self).__init__()
        self.type = 'MOEV6'

        # Feature processing layers
        self.feature_encoder = MLP(50 * 5, 512)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )
        self.map_encoder = MLP(49 * 256, 128)

        # CAV to agent feature downsizer.
        self.cav_to_agent = MLP(512 + 128 + 128, 512)

        # Trajectory decoder to decode a single trajectory embedding into a standard trajectory.
        self.trajectory_decoder = MLP(512, 50 * 5)

        # Estimator that for each mode and a BEV feature, estimate its gating factor
        self.gating_factor_estimators = nn.ModuleList([MLP(512 + 128 + 128, 1) for _ in range(6)])

        # Score Estimator.
        self.score_estimator = MLP(6 * 512, 6)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, center_objects_id_to_aggregate,
                map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores, center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav * num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav*num_modes, 50*5 -> num_center_objects_this_cav*num_modes, 512

            # Separate by agent.
            agent_feature_embeddings = [agent_feature_embedding for agent_feature_embedding in torch.unbind(cav_feature_embedding, dim=0)]
            center_objects_id = torch.from_numpy(center_objects_id).unsqueeze(-1).repeat(1, num_modes).flatten()  # num_center_objects_per_cav -> num_center_objects_per_cav*num_modes
            center_objects_id = center_objects_id.tolist()
            assert len(agent_feature_embeddings) == len(center_objects_id)

            for agent_feature_embedding, center_object_id in zip(agent_feature_embeddings, center_objects_id):
                predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [agent_feature_embedding]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        ego_map_features = map_polylines_feature[0]
        ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):
            agent_feature_embeddings = predictions_by_cav[center_object_id]  # list of (512)
            num_predictions_for_this_agent_with_modes = len(agent_feature_embeddings)

            # Stack predictions from different agents.
            agent_feature_embeddings = torch.stack(agent_feature_embeddings, dim=0)  # (num_predictions_for_this_agent*num_modes, 512)

            # Fuse BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 128)
            map_feature_embedding_this_agent = self.map_encoder((ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(num_predictions_for_this_agent_with_modes, 1) # (num_predictions_for_this_agent*num_modes, 128)
            cav_feature_embedding = torch.cat([agent_feature_embeddings,
                                               bev_feature_embedding_this_agent,
                                               map_feature_embedding_this_agent],
                                              dim=-1)  # (num_predictions_for_this_agent*num_modes, 512 + 128 + 128)

            # Estimate gating factor for each mode and aggregate.
            assert len(self.gating_factor_estimators) == num_modes
            agent_prediction_per_mode = []
            for mode_idx in range(num_modes):
                gating_factor = self.gating_factor_estimators[mode_idx](cav_feature_embedding)  # (num_predictions_for_this_agent*num_modes, 1)
                gating_factor = nn.functional.softmax(gating_factor, dim=0)

                # Combine the predictions.
                weights_cav_feature_embeddings = gating_factor * cav_feature_embedding  # (num_predictions_for_this_agent*num_modes, 512 + 128 + 128)
                aggregated_cav_feature_embedding = weights_cav_feature_embeddings.sum(dim=0)  # (512 + 128 + 128)
                aggregated_agent_feature_embedding = self.cav_to_agent(aggregated_cav_feature_embedding)
                agent_prediction_per_mode.append(aggregated_agent_feature_embedding)

            # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            combined_prediction = torch.stack(agent_prediction_per_mode, dim=0)  # (num_modes, 512)
            final_prediction_embeddings.append(combined_prediction)

            # Estimate the scores.
            scores = self.score_estimator(combined_prediction.view(-1))  # (num_modes)
            final_scores.append(scores)

        final_prediction_embeddings = torch.stack(final_prediction_embeddings, dim=0)  # (num_center_objects_first_cav, num_modes, 512)
        final_trajectories = self.trajectory_decoder(final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, 50 * 5)
        final_trajectories = final_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)
        final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, final_scores

class MotionAggregatorTransformer(nn.Module):
    """
    In this model we use a transformer to adjust trajectories with BEV feature and map.
    """
    def __init__(self):
        super(MotionAggregatorTransformer, self).__init__()
        self.type = 'Transformer'

        self.feature_encoder = MLP(50 * 5, 512)

        # Encoders for BEV and map features
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 128, 128)
        )
        # self.map_encoder = nn.Linear(49 * 256, 128)

        # Cross-attention modules
        self.bev_cross_attention = CrossAttention()
        # self.map_cross_attention = CrossAttention()

        # Transformer encoder for self-attention on trajectories
        # encoder_layers = TransformerEncoderLayer(768, nhead=8)
        encoder_layers = TransformerEncoderLayer(640, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=5)

        # Reduction decoder.
        self.query_embeddings = nn.Parameter(torch.randn(6, 768))
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=5)

        # Decoder for trajectories
        self.trajectory_decoder = MLP(512, 50 * 5)

        # Score Estimator.
        self.score_estimator = MLP(6 * 512, 6)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count, 
                center_objects_id_to_aggregate):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores, center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            flatten_cav_feature = cav_feature.view(num_center_objects_this_cav * num_modes, num_timestamps * num_states)
            cav_feature_embedding = self.feature_encoder(flatten_cav_feature)  # num_center_objects_this_cav*num_modes, 50*5 -> num_center_objects_this_cav*num_modes, 512

            # Separate by agent.
            agent_feature_embeddings = [agent_feature_embedding for agent_feature_embedding in torch.unbind(cav_feature_embedding, dim=0)]
            center_objects_id = torch.from_numpy(center_objects_id).unsqueeze(-1).repeat(1, num_modes).flatten()  # num_center_objects_per_cav -> num_center_objects_per_cav*num_modes
            center_objects_id = center_objects_id.tolist()
            assert len(agent_feature_embeddings) == len(center_objects_id)

            for agent_feature_embedding, center_object_id in zip(agent_feature_embeddings, center_objects_id):
                predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [agent_feature_embedding]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        # ego_map_features = map_polylines_feature[0]
        # ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):
            agent_feature_embeddings = predictions_by_cav[center_object_id]  # list of (512)
            num_predictions_for_this_agent_with_modes = len(agent_feature_embeddings)

            # Stack predictions from different agents.
            agent_feature_embeddings = torch.stack(agent_feature_embeddings, dim=0)  # (num_predictions_for_this_agent*num_modes, 512)

            # Encode BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 128)
            # map_feature_embedding_this_agent = self.map_encoder((ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(num_predictions_for_this_agent_with_modes, 1) # (num_predictions_for_this_agent*num_modes, 128)

            # Cross-attention with BEV and map features.
            enriched_traj_embedding = torch.concat([agent_feature_embeddings, bev_feature_embedding_this_agent], dim=-1)

            agent_feature_embeddings_transformed = self.transformer_encoder(enriched_traj_embedding)  # (num_predictions_for_this_agent*num_modes, 512)

            modes_of_embeddings = agent_feature_embeddings_transformed[:num_modes, :512].contiguous()  # (num_modes, 512)
            # Decoder: The query interacts with the source sequence
            # modes_of_embeddings = self.transformer_decoder(self.query_embeddings, agent_feature_embeddings_transformed)
            
            # Add to results.
            final_prediction_embeddings.append(modes_of_embeddings)  

            # Estimate the scores.
            scores = self.score_estimator(modes_of_embeddings.view(-1))  # (num_modes)
            final_scores.append(scores)

            # # Estimate gating factor for each mode and aggregate.
            # assert len(self.gating_factor_estimators) == num_modes
            # agent_prediction_per_mode = []
            # for mode_idx in range(num_modes):
            #     gating_factor = self.gating_factor_estimators[mode_idx](cav_feature_embedding)  # (num_predictions_for_this_agent*num_modes, 1)
            #     gating_factor = nn.functional.softmax(gating_factor, dim=0)

            #     # Combine the predictions.
            #     weights_agent_feature_embeddings = gating_factor * agent_feature_embeddings  # (num_predictions_for_this_agent*num_modes, 512)
            #     aggregated_agent_feature_embedding = weights_agent_feature_embeddings.sum(dim=0)  # (512)
            #     agent_prediction_per_mode.append(aggregated_agent_feature_embedding)

            # # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            # combined_prediction = torch.stack(agent_prediction_per_mode, dim=0)  # (num_modes, 512)
            # final_prediction_embeddings.append(combined_prediction)

            # # Estimate the scores.
            # scores = self.score_estimator(combined_prediction.view(-1))  # (num_modes)
            # final_scores.append(scores)

        final_prediction_embeddings = torch.stack(final_prediction_embeddings, dim=0)  # (num_center_objects_first_cav, num_modes, 512)
        final_trajectories = self.trajectory_decoder(final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, 50 * 5)
        final_trajectories = final_trajectories.view(ego_cav_traj_count, num_modes, num_timestamps, num_states)
        final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, final_scores


class NextTokenPredictionTransformer(nn.Module):
    def __init__(self, token_dim=256, hidden_dim=256, nhead=8, num_layers=1):
        super(NextTokenPredictionTransformer, self).__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(token_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, token_dim)

    def generate_square_subsequent_mask(self, sz):
        """ Generates an upper-triangular matrix of -inf, with zeros on diag. """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory):
        # tgt: (N, T, E) - Target sequence
        # memory: (N, S, E) - Encoder outputs (if any, or context)

        N, T, E = tgt.size()
        assert E == self.token_dim, f"Expected token dimension {self.token_dim}, got {E}"

        # Embed the target tokens
        tgt = self.embedding(tgt)  # (T, N, hidden_dim)

        # Create a causal mask
        tgt_mask = self.generate_square_subsequent_mask(T).to(tgt.device)  # (T, T)

        # Pass through the transformer decoder
        transformer_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # (T, N, hidden_dim)

        # Output layer to map back to token dimension
        output = self.output_layer(transformer_output)  # (T, N, token_dim)

        return output

class MotionAggregatorTransformerV2(nn.Module):
    """
    In this model we use a transformer to adjust trajectories with BEV feature and map.
    """

    def __init__(self):
        super(MotionAggregatorTransformerV2, self).__init__()
        self.type = 'TransformerV2'

        self.state_encoder = MLP(5, 32)

        # Encoders for BEV and map features
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 48 * 176, 128)
        )
        self.map_encoder = nn.Linear(49 * 256, 128)

        # Cross-attention modules
        self.bev_cross_attention = CrossAttention()
        self.map_cross_attention = CrossAttention()

        # Per timestamp transformer aggregator
        self.per_timestep_encoder_layer = nn.TransformerEncoderLayer(d_model=32+128+128, nhead=8)

        # Causality mask for decoder so that each token only attends to previous tokens.
        self.causality_mask = torch.triu(torch.ones(50, 50))

        # Transformer decoder based on next token prediction.
        self.trajectory_decoder_transformer = NextTokenPredictionTransformer(token_dim=32+128+128, hidden_dim=32+128+128, nhead=8, num_layers=4)

        # state decoder to decode a single trajectory embedding into a standard trajectory.
        self.state_decoder = MLP(32+128+128, 5)

        # Score Estimator.
        # self.score_estimator = MLP(6 * 50 * (32 + 128 + 128), 6)

    def forward(self, features_to_aggregate, pred_scores, bev_features, batch_sample_count,
                center_objects_id_to_aggregate, map_polylines_feature, map_valid_mask, map_polylines_center):
        """
        Args:
            features_to_aggregate: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            pred_scores: List of each CAV's trajectories to aggregate:
                each item has(num_center_objects_per_cav, num_modes, num_timestamps, 5)
                Note: the ego vehicle's prior prediction are always the first in the list.
            bev_features: bev_features from the ego (1, 256, 48, 176)
            batch_sample_count: int, how many trajs belong to the ego vehicle
            center_objects_id_to_aggregate: List of each CAV's prediction ids.
                Each item (num_center_objects_per_cav).
            map_polylines_feature: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 256)
            map_valid_mask: List of each CAV's map polylines. (num_center_objects_per_cav, 49)
            map_polylines_center: List of each CAV's map polylines. (num_center_objects_per_cav, 49, 3)
        Returns:
            output_trajectories: (num_center_objects_first_cav, num_modes, num_timestamps, 5)
            new_confidence_scores: (num_center_objects_first_cav, num_modes)
        """
        ego_cav_traj_count, num_modes, num_timestamps, num_states = features_to_aggregate[0].shape
        device = features_to_aggregate[0].device

        bev_features = bev_features.to(device)
        bev_feature_embedding = self.bev_encoder(bev_features)  # (1, 128)

        # Extract the trajectories from different CAV predictors into its owner.
        predictions_by_cav = {}  # cav_id -> list of predictions, each (num_modes, feature_dim)
        for cav_feature, pred_score, center_objects_id in zip(features_to_aggregate, pred_scores,
                                                              center_objects_id_to_aggregate):
            num_center_objects_this_cav = cav_feature.shape[0]

            # Encode trajectories.
            cav_feature_embeddings = self.state_encoder(
                cav_feature)  # num_center_objects_this_cav, num_modes, 50, 5 -> num_center_objects_this_cav, num_modes, 50, 32

            # Separate by agent.
            cav_feature_embeddings = [cav_feature_embedding for cav_feature_embedding in torch.unbind(cav_feature_embeddings, dim=0)]

            for cav_feature_embedding, center_object_id in zip(cav_feature_embeddings, center_objects_id):
                # Unpack by number of modes
                for mode_idx in range(num_modes):
                    predictions_by_cav[center_object_id] = predictions_by_cav.get(center_object_id, []) + [
                        cav_feature_embedding[mode_idx]]

        # Aggregate the predictions for each agent.
        final_prediction_embeddings = []
        final_scores = []
        ego_map_features = map_polylines_feature[0]
        ego_map_mask = map_valid_mask[0]
        for idx, center_object_id in enumerate(center_objects_id_to_aggregate[0].tolist()):  # Loop through with the ordering same as the ego. (Ego is always 0 indexed)
            cav_feature_embeddings = predictions_by_cav[center_object_id]  # list of (50, 32) trajectory embeddings
            num_predictions_for_this_agent_with_modes = len(cav_feature_embeddings)

            # Encode BEV and map.
            bev_feature_embedding_this_agent = bev_feature_embedding.repeat(num_predictions_for_this_agent_with_modes,
                                                                            1)  # (num_predictions_for_this_agent*num_modes, 128)
            map_feature_embedding_this_agent = self.map_encoder(
                (ego_map_features[idx] * ego_map_mask[idx].unsqueeze(-1)).view(-1)).repeat(
                num_predictions_for_this_agent_with_modes, 1)  # (num_predictions_for_this_agent*num_modes, 128)

            six_modes_per_timestep = []
            for t in range(num_timestamps):
                # Stack states at this time stamp.
                cav_timestap_feature_stacked = torch.stack([cav_feature_embedding[t] for cav_feature_embedding in cav_feature_embeddings], dim=0)  # (num_predictions_for_this_agent_with_modes, 32)

                # Cross-attention with BEV and map features.
                enriched_traj_embedding = torch.concat(
                    [cav_timestap_feature_stacked, bev_feature_embedding_this_agent, map_feature_embedding_this_agent], dim=-1) # (num_predictions_for_this_agent_with_modes, 32 + 128 + 128)

                # Stack predictions from different agents.
                agent_feature_embeddings = self.per_timestep_encoder_layer(enriched_traj_embedding)  # (num_predictions_for_this_agent_with_modes, 32 + 128 + 128)

                # Cut out the top 6 modes.
                top_modes = agent_feature_embeddings[:num_modes, :].contiguous()  # (num_modes, 32 + 128 + 128)
                six_modes_per_timestep.append(top_modes)

            # Stack all modes into leading dimension. This is all the 6 prediction embeddings we have for this agent.
            combined_prediction = torch.stack(six_modes_per_timestep, dim=1)  # (num_modes, num_timestamps, 32 + 128 + 128)

            # Auto-regressive transformer decoder.
            # agent_feature_embeddings_transformed = self.trajectory_decoder_transformer(combined_prediction, combined_prediction)  # (num_modes, num_timestamps, 32 + 128 + 128)

            final_prediction_embeddings.append(combined_prediction)

            # Estimate the scores.
            # scores = self.score_estimator(agent_feature_embeddings_transformed.view(-1))  # (num_modes)
            # final_scores.append(scores)


        final_prediction_embeddings = torch.stack(final_prediction_embeddings,
                                                  dim=0)  # (num_center_objects_first_cav, num_modes, num_timestamps, 32 + 128 + 128)
        final_trajectories = self.state_decoder(
            final_prediction_embeddings)  # (num_center_objects_first_cav, num_modes, num_timestamps, 5)
        # final_scores = torch.stack(final_scores, dim=0)  # (num_center_objects_first_cav, num_modes)

        return final_trajectories, pred_scores[0]

class MotionTransformerWithMultiEgoAggregation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        self.motion_transformer = MotionTransformer(config)

        if self.model_cfg.MOTION_AGGREGATOR.TYPE == 'None':
            pass
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MLP':
            self.motion_aggregator = MotionAggregatorMLP()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MLPV2':
            self.motion_aggregator = MotionAggregatorMLPV2()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'GCN':
            self.motion_aggregator = MotionAggregatorGCN()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MOE':
            self.motion_aggregator = MotionAggregatorMOE()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MOEV2':
            self.motion_aggregator = MotionAggregatorMOEV2()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MOEV3':
            self.motion_aggregator = MotionAggregatorMOEV3()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MOEV4':
            self.motion_aggregator = MotionAggregatorMOEV4()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MOEV5':
            self.motion_aggregator = MotionAggregatorMOEV5()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'MOEV6':
            self.motion_aggregator = MotionAggregatorMOEV6()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'Transformer':
            self.motion_aggregator = MotionAggregatorTransformer()
        elif self.model_cfg.MOTION_AGGREGATOR.TYPE == 'TransformerV2':
            self.motion_aggregator = MotionAggregatorTransformerV2()
        else:
            raise RuntimeError(f"Unknown motion aggregator type: {self.model_cfg.MOTION_AGGREGATOR.TYPE}")

    def forward(self, batch_dict):
        """
        Args:
            batch_dict: As defined in dataset.py::collate_batch, this is a dict of features.
            The keys are scenario_id, obj_trajs etc and the values are a batched list of values.
            batch_dict: {
                num_cavs: int
                input_dict: {
                    ... Input dict is a collated set of CAV features in a scene.
                }
                batch_sample_count: list of track_index_to_predict
                is_ego_one_hot: list of 0, 1 indicating where the index corresponds to ego is 1.
                fused_bev_features:... Fused BEV features is a collated batch of individual CAV features.
            }
        """
        # Batch all cavs in a scene, then run motion transformer.
        # Note: mtr_outputs['pred_trajs'] = (num_center_objects, num_modes, num_timestamps, 7)
        if self.training:
            mtr_outputs, loss, tb_dict, disp_dict = self.motion_transformer(batch_dict)
            mtr_outputs['pre_aggregation_pred_trajs'] = mtr_outputs['pred_trajs'].clone().detach()
            if self.model_cfg.MOTION_AGGREGATOR.TYPE == 'None':
                return loss, tb_dict, disp_dict
        else:
            mtr_outputs = self.motion_transformer(batch_dict)
            mtr_outputs['pre_aggregation_pred_trajs'] = mtr_outputs['pred_trajs'].clone().detach()
            if self.model_cfg.MOTION_AGGREGATOR.TYPE == 'None':
                return mtr_outputs

        # Extract the outputs for each cav.
        trajs_per_cav = []
        pred_scores_per_cav = []
        center_objects_id_per_cav = []
        map_polylines_feature_per_cav = []
        map_valid_mask_per_cav = []
        map_polylines_center_per_cav = []

        start_obj_idx = 0
        for cav_idx in range(batch_dict['num_cavs']):
            trajs_per_cav.append(mtr_outputs['pred_trajs'][start_obj_idx:start_obj_idx + batch_dict['batch_sample_count'][cav_idx]])
            pred_scores_per_cav.append(mtr_outputs['pred_scores'][start_obj_idx:start_obj_idx + batch_dict['batch_sample_count'][cav_idx]])
            center_objects_id_per_cav.append(batch_dict['input_dict']['center_objects_id'][start_obj_idx:start_obj_idx + batch_dict['batch_sample_count'][cav_idx]])
            # map_polylines_feature_per_cav.append(batch_dict['map_feature'][start_obj_idx:start_obj_idx + batch_dict['batch_sample_count'][cav_idx]])
            # map_valid_mask_per_cav.append(batch_dict['map_mask'][start_obj_idx:start_obj_idx + batch_dict['batch_sample_count'][cav_idx]])
            # map_polylines_center_per_cav.append(batch_dict['map_pos'][start_obj_idx:start_obj_idx + batch_dict['batch_sample_count'][cav_idx]])
            start_obj_idx += batch_dict['batch_sample_count'][cav_idx]

        # For each CAV, run motion aggregator.
        aggregated_pred_trajs_per_cav = []
        aggregated_pred_scores_per_cav = []
        for ego_cav_idx in range(batch_dict['num_cavs']):
            ego_traj_shape = None
            ego_pred_score_shape = None

            features_to_aggregate = []
            pred_scores_to_aggregate = []
            center_objects_id_to_aggregate = []
            map_polylines_feature_to_aggregate = []
            map_valid_mask_to_aggregate = []
            map_polylines_center_to_aggregate = []

            for other_cav_idx in range(batch_dict['num_cavs']):
                if ego_cav_idx == other_cav_idx:
                    features_to_aggregate.insert(0, trajs_per_cav[other_cav_idx])
                    pred_scores_to_aggregate.insert(0, pred_scores_per_cav[other_cav_idx])
                    center_objects_id_to_aggregate.insert(0, center_objects_id_per_cav[other_cav_idx])
                    # map_polylines_feature_to_aggregate.insert(0, map_polylines_feature_per_cav[other_cav_idx])
                    # map_valid_mask_to_aggregate.insert(0, map_valid_mask_per_cav[other_cav_idx])
                    # map_polylines_center_to_aggregate.insert(0, map_polylines_center_per_cav[other_cav_idx])

                    ego_traj_shape = trajs_per_cav[other_cav_idx].shape
                    ego_pred_score_shape = pred_scores_per_cav[other_cav_idx].shape
                else:
                    truncated_trajs = torch.zeros_like(trajs_per_cav[other_cav_idx])
                    truncated_trajs[:, :, 1:, :] = trajs_per_cav[other_cav_idx][:, :, 1:, :]
                    features_to_aggregate.append(truncated_trajs)
                    pred_scores_to_aggregate.append(pred_scores_per_cav[other_cav_idx])
                    center_objects_id_to_aggregate.append(center_objects_id_per_cav[other_cav_idx])
                    # map_polylines_feature_to_aggregate.append(map_polylines_feature_per_cav[other_cav_idx])
                    # map_valid_mask_to_aggregate.append(map_valid_mask_per_cav[other_cav_idx])
                    # map_polylines_center_to_aggregate.append(map_polylines_center_per_cav[other_cav_idx])

            aggregated_trajs, aggregated_pred_scores = self.motion_aggregator(features_to_aggregate,
                                                                              pred_scores_to_aggregate,
                                                                              batch_dict['input_dict']['fused_feature'][ego_cav_idx],# (1, 256, 48, 176)
                                                                              batch_dict['batch_sample_count'][ego_cav_idx],
                                                                              center_objects_id_to_aggregate)
                                                                            #   map_polylines_feature_to_aggregate,
                                                                            #   map_valid_mask_to_aggregate,
                                                                            #   map_polylines_center_to_aggregate)

            assert aggregated_trajs.shape == ego_traj_shape
            assert aggregated_pred_scores.shape == ego_pred_score_shape

            aggregated_pred_trajs_per_cav.append(aggregated_trajs)
            aggregated_pred_scores_per_cav.append(aggregated_pred_scores)

        mtr_outputs['aggregated_pred_trajs_list'] = aggregated_pred_trajs_per_cav
        mtr_outputs['aggregated_pred_scores_list'] = aggregated_pred_scores_per_cav

        aggregated_pred_trajs = torch.cat(aggregated_pred_trajs_per_cav, dim=0)
        aggregated_pred_scores = torch.cat(aggregated_pred_scores_per_cav, dim=0)

        mtr_outputs['pred_trajs'] = aggregated_pred_trajs
        mtr_outputs['pred_scores'] = aggregated_pred_scores

        if self.training:
            # for aggregated_pred_trajs, aggregated_pred_scores in zip(aggregated_pred_trajs_per_cav, aggregated_pred_scores_per_cav):
            loss = self.get_individual_cav_loss(aggregated_pred_trajs, aggregated_pred_scores)

            tb_dict.update({'loss': loss.item()})
            disp_dict.update({'loss': loss.item()})
            return loss, tb_dict, disp_dict

        return mtr_outputs

    def get_individual_cav_loss(self, pred_trajs, pred_scores):
        """
        Args:
            aggregated_traj_pred: (N, M, T, 7)
        """

        center_gt_trajs = self.motion_transformer.motion_decoder.forward_ret_dict['center_gt_trajs'].cuda()
        center_gt_trajs_mask = self.motion_transformer.motion_decoder.forward_ret_dict['center_gt_trajs_mask'].cuda()
        # center_gt_final_valid_idx = self.motion_transformer.motion_decoder.forward_ret_dict['center_gt_final_valid_idx'].long()
        assert center_gt_trajs.shape[-1] == 2

        # pred_list = self.motion_transformer.motion_decoder.forward_ret_dict['pred_list']
        # intention_points = self.motion_transformer.motion_decoder.forward_ret_dict['intention_points']  # (num_center_objects, num_query, 2)

        num_center_objects = center_gt_trajs.shape[0]
        # center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2]  # (num_center_objects, 2)

        # dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
        # center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)

        total_loss = 0

        assert pred_trajs.shape[-1] == 5
        pred_trajs_gmm = pred_trajs[:, :, :, 0:5]

        # print(f'pred_trajs_gmm.shape: {pred_trajs_gmm.shape} pred_scores.shape: {pred_scores.shape}')
        loss_reg_gmm, center_gt_positive_idx = loss_utils.nll_loss_gmm_direct(
            pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,
            gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,
            pre_nearest_mode_idxs=None,
            timestamp_loss_weight=None, use_square_gmm=False,
        )

        loss_cls = F.cross_entropy(input=pred_scores, target=center_gt_positive_idx, reduction='none')

        # total loss
        weight_cls = self.motion_transformer.motion_decoder.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)
        weight_reg = self.motion_transformer.motion_decoder.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)

        layer_loss = loss_reg_gmm * weight_reg + loss_cls.sum(dim=-1) * weight_cls
        layer_loss = layer_loss.mean()
        total_loss += layer_loss
        return total_loss

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        # logger.info('==> Done')
        logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'The number of missing keys: {len(missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch

