import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
    This class is designed to scatter the features of point pillars back into a 2D grid (BEV: Bird's Eye View),
    which is a common step in PointPillars, a popular method for processing point clouds.
    """
    def __init__(self, model_cfg):
        """
        The constructor initializes the module using a configuration dictionary model_cfg.
        It extracts the number of BEV features (num_bev_features) and grid size (nx, ny, nz) from model_cfg.
        The assertion assert self.nz == 1 ensures that the grid is essentially a 2D grid (as is typical for BEV representation).
        """
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']
        assert self.nz == 1

    def forward(self, batch_dict):
        """
        The forward method takes batch_dict, a dictionary containing batched data, including pillar_features (features extracted from point pillars) and voxel_coords (coordinates of the voxels).
        The method processes each item in the batch separately.
        For each batch, it creates a zero-initialized tensor spatial_feature of size [num_bev_features, nz * nx * ny]. Since nz is 1, this is essentially a 2D grid with num_bev_features channels.
        It then selects the pillars and their corresponding coordinates for the current batch (batch_idx).
        The method computes linear indices from the voxel coordinates and uses these indices to scatter the pillar features into the 2D grid (spatial_feature).
        Each scattered feature is appended to batch_spatial_features.
        After processing all batches, batch_spatial_features is reshaped and added to batch_dict under the key 'spatial_features'.
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict[
            'voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]

            indices = this_coords[:, 1] + \
                      this_coords[:, 2] * self.nx + \
                      this_coords[:, 3]
            indices = indices.type(torch.long)

            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = \
            torch.stack(batch_spatial_features, 0)
        batch_spatial_features = \
            batch_spatial_features.view(batch_size, self.num_bev_features *
                                        self.nz, self.ny, self.nx)

        batch_dict['spatial_features'] = batch_spatial_features

        return batch_dict

