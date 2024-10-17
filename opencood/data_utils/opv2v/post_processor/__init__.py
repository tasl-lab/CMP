# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.opv2v.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.opv2v.post_processor.bev_postprocessor import BevPostprocessor
from opencood.data_utils.opv2v.post_processor.ciassd_postprocessor import CiassdPostprocessor
from opencood.data_utils.opv2v.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor
from opencood.data_utils.opv2v.post_processor.motr_postprocessor import MOTRPostprocessor

__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'BevPostprocessor': BevPostprocessor,
    'CiassdPostprocessor': CiassdPostprocessor,
    'FpvrcnnPostprocessor': FpvrcnnPostprocessor,
    'MOTRPostprocessor': MOTRPostprocessor
}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['VoxelPostprocessor', 'BevPostprocessor', 'CiassdPostprocessor', 'FpvrcnnPostprocessor', 'MOTRPostprocessor']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        train=train
    )

    return anchor_generator
