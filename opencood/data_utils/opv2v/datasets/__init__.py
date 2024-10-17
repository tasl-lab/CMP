# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.opv2v.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.opv2v.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.opv2v.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.opv2v.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
from opencood.data_utils.opv2v.datasets.intermediate_fusion_dataset_multiframes import IntermediateFusionDatasetMultiFrame
from opencood.data_utils.opv2v.datasets.intermediate_fusion_dataset_multi_ego import IntermediateFusionDatasetMultiEgo
from opencood.data_utils.opv2v.datasets.no_fusion_multi_ego import NoFusionDatasetMultiEgo

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDatasetV2': IntermediateFusionDatasetV2,
    'IntermediateFusionDatasetMultiFrame': IntermediateFusionDatasetMultiFrame,
    'IntermediateFusionDatasetMultiEgo': IntermediateFusionDatasetMultiEgo,
    'NoFusionDatasetMultiEgo': NoFusionDatasetMultiEgo
}

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
# The communication range for cavs
COM_RANGE = 70


def build_opv2v_dataset(dataset_cfg, visualize=False, train=True):
    print('Building Dataset from OPV2V')
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/opv2v/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset', 'IntermediateFusionDatasetV2',
                            'IntermediateFusionDatasetMultiFrame',
                            'IntermediateFusionDatasetMultiEgo', 'NoFusionDatasetMultiEgo'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
