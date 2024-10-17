import argparse
import math
import sys

import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.fuse_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

sys.path.append("./MOTR")
from MOTR.models.deformable_detr import MLP
from MOTR.models.deformable_transformer_plus import DeformableTransformer
from MOTR.models.matcher import HungarianMatcher
from MOTR.models.motr import (ClipMatcher, RuntimeTrackerBase,
                              TrackerPostProcess, _get_clones)
from MOTR.models.position_encoding import (PositionEmbeddingLearned,
                                           PositionEmbeddingSine)
from MOTR.models.qim import QueryInteractionModule
from MOTR.models.structures.instances import Instances
from MOTR.util.misc import inverse_sigmoid, nested_tensor_from_tensor_list
from MOTR.util.tool import load_model


class PointPillarCoBEVTMOTR(nn.Module):
    def __init__(self, args):
        super(PointPillarCoBEVTMOTR, self).__init__()

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

        self.fusion_net = SwapFusionEncoder(args['fax_fusion'])

        # self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
        #                           kernel_size=1)
        # self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
        #                           kernel_size=1)

        deformable_transformer_args = args['deformable_transformer']
        self.transformer = DeformableTransformer(
            d_model=deformable_transformer_args['hidden_dim'],
            nhead=deformable_transformer_args['nheads'],
            num_encoder_layers=deformable_transformer_args['enc_layers'],
            num_decoder_layers=deformable_transformer_args['dec_layers'],
            dim_feedforward=deformable_transformer_args['dim_feedforward'],
            dropout=deformable_transformer_args['dropout'],
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=deformable_transformer_args['num_feature_levels'],
            dec_n_points=deformable_transformer_args['dec_n_points'],
            enc_n_points=deformable_transformer_args['enc_n_points'],
            two_stage=deformable_transformer_args['two_stage'],
            two_stage_num_proposals=deformable_transformer_args['num_queries'],
            decoder_self_cross=not deformable_transformer_args['decoder_cross_self'],
            sigmoid_attn=deformable_transformer_args['sigmoid_attn'],
            extra_track_attn=deformable_transformer_args['extra_track_attn'],
        )

        model_path = deformable_transformer_args['pretrained_path']
        model_without_ddp = load_model(self.transformer, model_path)

        query_iteraction_module_args = args['query_iteraction_module']
        class params:
            def __init__(self):
                self.random_drop = query_iteraction_module_args['random_drop']
                self.fp_ratio = query_iteraction_module_args['fp_ratio']
                self.update_query_pos = query_iteraction_module_args['update_query_pos']
                self.merger_dropout = query_iteraction_module_args['merger_dropout']
        qim_args = params()
        self.query_interaction_layer = QueryInteractionModule(
            qim_args, 
            dim_in = query_iteraction_module_args['dim_in'],
            hidden_dim= query_iteraction_module_args['hidden_dim'],
            dim_out=query_iteraction_module_args['dim_out']
        )

        matcher_args = args['matcher']
        self.matcher = HungarianMatcher(
            cost_class = matcher_args['cost_class'],
            cost_bbox = matcher_args['cost_bbox'],
            cost_giou = matcher_args['cost_giou']
        )

        self.queue_length = args['queue_length']
        self.weight_dict = {}
        for i in range(self.queue_length):
            self.weight_dict.update({"frame_{}_loss_ce".format(i): matcher_args['cost_class'],
                                'frame_{}_loss_bbox'.format(i): matcher_args['cost_bbox'],
                                'frame_{}_loss_giou'.format(i): matcher_args['cost_giou'],
                                })

        if args['aux_loss']:
            for i in range(self.queue_length):
                for j in range(deformable_transformer_args['dec_layers'] - 1):
                    self.weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): matcher_args['cost_class'],
                                        'frame_{}_aux{}_loss_bbox'.format(i, j): matcher_args['cost_bbox'],
                                        'frame_{}_aux{}_loss_giou'.format(i, j): matcher_args['cost_giou'],
                                        })
        
        num_classes = 1
        self.criterion = ClipMatcher(num_classes, matcher=self.matcher, weight_dict=self.weight_dict, losses=args['losses'])

        #MOTR
        self.num_queries = deformable_transformer_args['num_queries']
        hidden_dim = self.transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 7, 3)
        self.num_feature_levels = deformable_transformer_args['num_feature_levels']
        self.use_checkpoint = args['use_checkpoint']
        self.two_stage = deformable_transformer_args['two_stage']
        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim * 2)
        self.aux_loss = args['aux_loss']
        self.with_box_refine = args['with_box_refine']

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.xavier_uniform_(self.bbox_embed.layers[-1].weight.data, gain=1)
        num_pred = (self.transformer.decoder.num_layers + 1) if self.two_stage else self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            # nn.init.xavier_uniform_(self.bbox_embed[0].layers[-1].bias.data[2:], gain=1)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.xavier_uniform_(self.bbox_embed.layers[-1].bias.data[2:], gain=1)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.memory_bank = args['memory_bank']
        self.mem_bank_len = 0 if self.memory_bank is None else self.memory_bank.max_his_length

        N_steps = args['position_embedding']['hidden_dim']
        if args['position_embedding']['type'] in ('v2', 'sine'):
            self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif args['position_embedding']['type'] in ('v3', 'learned'):
            self.position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {args['position_embedding']['type']}")

        self.prob_threshold = args['prob_threshold']
        self.area_threshold = args['area_threshold']

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

    def _convert_to_instance(self, targets: dict):
        gt_instances = Instances((0, 0))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        track_instances.ref_pts = self.transformer.reference_points(self.query_embed.weight[:, :dim // 2])
        track_instances.query_pos = self.query_embed.weight
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 7), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embed.weight.device)

    def _forward_single_timestamp(self, current_fused_feature, track_instances: Instances):
        bs, dim, h, w = current_fused_feature.shape
        src = current_fused_feature # target: bs, dim, h, w
        mask = torch.full((bs, h, w), False, dtype=torch.bool).to(src.device)# target: bs, h, w
        assert mask is not None
        place_holder_src = torch.zeros((dim, h, w)).to(src.device)
        src4pe = nested_tensor_from_tensor_list([place_holder_src])
        pos = self.position_embedding(src4pe)
        srcs = [src]
        masks = [mask]
        pos = [pos]

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            elif reference.shape[-1] == 7:
                tmp[..., :2] += reference[:, :, 0:2]
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[5]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, is_last):
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.query_interaction_layer(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res

    def forward(self, data_dict):
        fused_features = []
        gt_instances = []

        # def plot_fig(idx, cur_tensor):
        #     import matplotlib.pyplot as plt
        #     cur_tensor = cur_tensor[:, :2]
        #     cur_tensor = cur_tensor.cpu().detach().numpy()
        #     plt.scatter(cur_tensor[:,0], cur_tensor[:,1])
        #     plt.savefig(f"img_{idx}.png")
                
        for timestamp in range(self.queue_length):
            object_bbx_center = data_dict['object_bbx_center'][timestamp, ...]
            object_bbx_mask = data_dict['object_bbx_mask'][timestamp, ...]
            valid_object_bbx = object_bbx_center[torch.where(object_bbx_mask==1)]
            # plot_fig(timestamp, valid_object_bbx)
            obj_id = torch.tensor(data_dict['object_ids'][timestamp]).to(object_bbx_center.device)
            label = torch.zeros(len(obj_id)).to(object_bbx_center.device)
            area = valid_object_bbx[:, 3] * valid_object_bbx[:, 4] # x, y, z, h, w, l, yaw
            target = {
                'boxes': valid_object_bbx.float(),
                'labels': label.long(),
                'obj_ids': obj_id.float(),
                'area': area.float()
            }
            gt_instance = self._convert_to_instance(target)
            gt_instances.append(gt_instance)

            voxel_features = data_dict['processed_lidar'][timestamp]['voxel_features']
            voxel_coords = data_dict['processed_lidar'][timestamp]['voxel_coords']
            voxel_num_points = data_dict['processed_lidar'][timestamp]['voxel_num_points']
            record_len = data_dict['record_len']
            spatial_correction_matrix = data_dict['spatial_correction_matrix'][timestamp, ...]

            batch_dict = {'voxel_features': voxel_features,
                        'voxel_coords': voxel_coords,
                        'voxel_num_points': voxel_num_points,
                        'record_len': record_len}
            # n, 4 -> n, c
            batch_dict = self.pillar_vfe(batch_dict)
            # n, c -> N, C, H, W
            batch_dict = self.scatter(batch_dict)
            batch_dict = self.backbone(batch_dict)

            spatial_features_2d = batch_dict['spatial_features_2d']
            # downsample feature to reduce memory
            if self.shrink_flag:
                spatial_features_2d = self.shrink_conv(spatial_features_2d)
            # compressor
            if self.compression:
                spatial_features_2d = self.naive_compressor(spatial_features_2d)

            # N, C, H, W -> B,  L, C, H, W
            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            self.max_cav)
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            com_mask = repeat(com_mask,
                            'b h w c l -> b (h new_h) (w new_w) c l',
                            new_h=regroup_feature.shape[3],
                            new_w=regroup_feature.shape[4])

            fused_feature = self.fusion_net(regroup_feature, com_mask)
            fused_features.append(fused_feature)

        if self.training:
            self.criterion.initialize_for_single_clip(gt_instances)

        output_dict = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        track_instances = self._generate_empty_tracks()
        keys = list(track_instances._fields.keys())

        for timestamp in range(self.queue_length):
            # if not self.training:
            #     if track_instances is not None:
            #         track_instances.remove('boxes')
            #         track_instances.remove('labels')
            is_last = timestamp == self.queue_length - 1 if self.training else False
            current_fused_feature = fused_features[timestamp]
            res = self._forward_single_timestamp(current_fused_feature, track_instances)
            res = self._post_process_single_image(res, track_instances, is_last)
            track_instances = res['track_instances']
            output_dict['pred_logits'].append(res['pred_logits'])
            output_dict['pred_boxes'].append(res['pred_boxes'])

            if not self.training:
                self.post_process(track_instances, None)
                keep = track_instances.scores > self.prob_threshold
                track_instances = track_instances[keep]
                # if len(track_instances) != 0:
                #     areas = track_instances.boxes[:, 3] * track_instances.boxes[:, 4]
                #     keep = area > self.area_threshold
                #     track_instances = track_instances[keep]
                output_dict['pred_logits'][timestamp] = track_instances.pred_logits.unsqueeze(dim=0) #bs != 1 ?
                output_dict['pred_boxes'][timestamp] = track_instances.pred_boxes.unsqueeze(dim=0)
        
        if not self.training:
            output_dict['track_instances'] = track_instances
        else:
            output_dict['losses_dict'] = self.criterion.losses_dict

        if self.training:
            loss_dict = self.criterion(output_dict, data_dict)
            weight_dict = self.criterion.weight_dict
            total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            reg_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict and 'bbox' in k)
            conf_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict and 'ce' in k)
            output_dict['total_loss'] = total_loss
            output_dict['reg_loss'] = reg_loss
            output_dict['conf_loss'] = conf_loss

        # psm = self.cls_head(fused_feature) #1, 2, 48, 176
        # rm = self.reg_head(fused_feature) #1, 14, 48, 176

        # output_dict = {'psm': psm,
        #                'rm': rm}

        return output_dict
