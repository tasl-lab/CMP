import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.fuse_modules.graph_sage_net import GraphSageNet


class PointPillarCoBEVT(nn.Module):
    def __init__(self, args):
        super(PointPillarCoBEVT, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.graph_sage_net = GraphSageNet(256, 256, 256)
        self.post_graph_cls_head = nn.Linear(5 * 256,  2*48*176)
        self.post_graph_reg_head = nn.Linear(5 * 256, 14*48*176)

        self.fusion_net = SwapFusionEncoder(args['fax_fusion'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)  # pillar_features: 92683, 64
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)  # spatial_features: 12, 64, 192, 704
        batch_dict = self.backbone(batch_dict)  # spatial_features_2d: 12, 384, 96, 352

        spatial_features_2d = batch_dict['spatial_features_2d']  # spatial_features_2d: 12, 384, 96, 352
        
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)  # 12, 256, 48, 176
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # N, C, H, W -> B, L=5, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # regroup_feature: 4, 5, 256, 48, 176
        # mask: 4, 5

        # GCN
        # outputs = self.graph_sage_net(regroup_feature, mask)  # batch_size, car, output_dim(256)
        # outputs = outputs.reshape(outputs.shape[0], -1)  # batch_size, car * output_dim(256)
        # cls_out = self.post_graph_cls_head(outputs)  # batch_size, car * anchor_number(2)
        # reg_out = self.post_graph_reg_head(outputs)  # batch_size, car * anchor_number(14)
        # psm = cls_out.reshape(outputs.shape[0], 2, 48, 176)
        # rm  = reg_out.reshape(outputs.shape[0], 14, 48, 176)
        # End of GCN

        # CoBEVT FusionNet: FuseBEVT
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 4, 1, 1, 1, 5
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])  # 4, 48, 176, 1, 5

        fused_feature = self.fusion_net(regroup_feature, com_mask)  # 4, 256, 48, 176

        psm = self.cls_head(fused_feature)  # 4, 2, 48, 176
        rm = self.reg_head(fused_feature)  # 4, 14, 48, 176
        # End of CoBEVT FusionNet

        output_dict = {'psm': psm,
                       'rm': rm,
                       'spatial_features_2d': spatial_features_2d.clone().detach(),
                       'fused_feature': fused_feature.clone().detach()}

        return output_dict
