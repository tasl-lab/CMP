from opencood.data_utils.v2v4real.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.v2v4real.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.v2v4real.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.v2v4real.datasets.intermediate_fusion_dataset_multi_ego import IntermediateFusionDatasetMultiEgo
from opencood.data_utils.v2v4real.datasets.no_fusion_multi_ego import NoFusionDatasetMultiEgo

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDatasetMultiEgo': IntermediateFusionDatasetMultiEgo,
    'NoFusionDatasetMultiEgo': NoFusionDatasetMultiEgo
}

# the final range for evaluation
GT_RANGE = [-100, -40, -5, 100, 40, 3]
# The communication range for cavs
COM_RANGE = 70

def build_v2v4real_dataset(dataset_cfg, visualize=False, train=True, isSim=False):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset', 'IntermediateFusionDatasetMultiEgo', 'NoFusionDatasetMultiEgo'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        isSim=isSim
    )

    return dataset
