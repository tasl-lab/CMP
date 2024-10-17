"""
Dataset class for intermediate fusion
"""
import random
import math
import warnings
from collections import OrderedDict

import os
import math
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset

import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.v2v4real.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
import opencood.data_utils.v2v4real.datasets
import opencood.data_utils.v2v4real.post_processor as post_processor
from opencood.utils import box_utils_v2v4real as box_utils
from opencood.data_utils.v2v4real.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, dist_two_pose

class NoFusionDatasetMultiEgo(Dataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True, isSim=False):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.isSim = isSim

        self.augment_config = params['data_augment']
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        self.idx2scenrio = {}

        # if the training/testing include noisy setting
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']

        if 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']

        print("Built NoFusionDatasetMultiEgo for " + root_dir)

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        # Old:
        # Structure: {scenario_id :
        #                   {cav_1 :
        #                       {timestamp1 :
        #                           {yaml: path,
        #                           lidar: path,
        #                           cameras:list of path}
        #                        }
        #                   }
        #             }

        # New:
        # Structure: {scenario_id :
        #                   {ego_cav_id :
        #                       {cav_1 :
        #                           {timestamp1 :
        #                            {yaml: path,
        #                            lidar: path,
        #                               cameras:list of path}
        #                           }
        #                       }
        #                   }
        #             }
        self.scenario_database = OrderedDict()  # save all scenarios
        self.records = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})
            self.idx2scenrio[i] = scenario_folder
            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data in a scenario
            for ego_cav_id in cav_list:
                self.scenario_database[i][ego_cav_id] = OrderedDict()

                for cav_id in cav_list:

                    self.scenario_database[i][ego_cav_id][cav_id] = OrderedDict()

                    # save all yaml files to the dictionary
                    cav_path = os.path.join(scenario_folder, cav_id)

                    yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' \
                            not in x and 'camera_gt' not in x])

                    timestamps = self.extract_timestamps(yaml_files)

                    for (t, timestamp) in enumerate(timestamps):
                        self.scenario_database[i][ego_cav_id][cav_id][timestamp] = \
                            OrderedDict()

                        yaml_file = os.path.join(cav_path,
                                                 timestamp + '.yaml')
                        lidar_file = os.path.join(cav_path,
                                                  timestamp + '.pcd')
                        # camera_files = self.load_camera_files(cav_path, timestamp)

                        self.scenario_database[i][ego_cav_id][cav_id][timestamp]['yaml'] = \
                            yaml_file
                        self.scenario_database[i][ego_cav_id][cav_id][timestamp]['lidar'] = \
                            lidar_file
                        # self.scenario_database[i][ego_cav_id][cav_id][timestamp]['camera0'] = \
                        #     camera_files

                        self.records.append({
                            'scenario_idx': i,
                            'timestamp_idx': t,         # int
                            'timestamp_key': timestamp,  # string
                            'ego_cav_id': ego_cav_id
                        })
                    # Assume all cavs will have the same timestamps length. Thus
                    # we only need to calculate for the first vehicle in the
                    # scene.
                    if cav_id == ego_cav_id:
                        self.scenario_database[i][ego_cav_id][cav_id]['ego'] = True
                    else:
                        self.scenario_database[i][ego_cav_id][cav_id]['ego'] = False

        # Convert scenario_database to be ordered by scenario, who_is_ego, timestamp.
        unique_records = {}
        for record in self.records:
            key = (record['scenario_idx'], record['timestamp_idx'], record['ego_cav_id'])
            if key not in unique_records:
                unique_records[key] = record

        # Extract values and sort
        deduplicated_sorted_records = sorted(unique_records.values(), key=lambda x: (
            x['scenario_idx'], x['ego_cav_id'], x['timestamp_idx']))
        self.records = deduplicated_sorted_records

        print(f'Preparing {len(self.records)} records for {len(self.scenario_database)} scenarios')
        # Child Class Contents ---------------

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if 'proj_first' in params['fusion']['args'] and \
            not params['fusion']['args']['proj_first']:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in \
            params['fusion']['args'] else \
            params['fusion']['args']['cur_ego_pose_flag']

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)
    
    def generate_augment(self):
        flip = [None, None, None]
        noise_rotation = None
        noise_scale = None

        for aug_ele in self.augment_config:
            # for intermediate fusion only
            if 'random_world_rotation' in aug_ele['NAME']:
                rot_range = \
                    aug_ele['WORLD_ROT_ANGLE']
                if not isinstance(rot_range, list):
                    rot_range = [-rot_range, rot_range]
                noise_rotation = np.random.uniform(rot_range[0],
                                                        rot_range[1])

            if 'random_world_flip' in aug_ele['NAME']:
                for i, cur_axis in enumerate(aug_ele['ALONG_AXIS_LIST']):
                    enable = np.random.choice([False, True], replace=False,
                                              p=[0.5, 0.5])
                    flip[i] = enable

            if 'random_world_scaling' in aug_ele['NAME']:
                scale_range = \
                    aug_ele['WORLD_SCALE_RANGE']
                noise_scale = \
                    np.random.uniform(scale_range[0], scale_range[1])

        return flip, noise_rotation, noise_scale

    def __len__(self):
        """
        Return the length of the dataset. Now defined as num of scenarios * num of ego cavs * num of cavs * timestamps

        """
        return len(self.records)

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # Find the scenario index by looping over the cumulative length of records.
        scenario_index = self.records[idx]['scenario_idx']
        ego_cav_id     = self.records[idx]['ego_cav_id']
        timestamp_idx  = self.records[idx]['timestamp_idx']
        timestamp_key  = self.records[idx]['timestamp_key']

        scenario_data_on_ego_cav = self.scenario_database[scenario_index][ego_cav_id]

        # calculate distance to ego for each cav
        ego_cav_content = self.calc_dist_to_ego(scenario_data_on_ego_cav, timestamp_key, ego_cav_id, scenario_index)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_data_on_ego_cav.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            time_delay_frames = self.time_delay_calculation(cav_content['ego'])

            timestamp_idx_delayed = max(0, timestamp_idx - time_delay_frames)
            timestamp_key_delayed = list(cav_content.keys())[timestamp_idx_delayed]

            # add time delay to vehicle parameters
            data[cav_id]['time_delay_frames'] = time_delay_frames
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content,
                                                    ego_cav_content,
                                                    timestamp_key,
                                                    timestamp_key_delayed,
                                                    cur_ego_pose_flag)

            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key_delayed]['lidar'])
            data[cav_id]['scenario_name'] = self.idx2scenrio[scenario_index]
            data[cav_id]['timestamp_idx'] = timestamp_idx
            data[cav_id]['timestamp_key'] = timestamp_key
            data[cav_id]['ego_cav_id'] = int(ego_cav_id)
            data[cav_id]['cav_id'] = int(cav_id)
        return data

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    def calc_dist_to_ego(self, scenario_data_on_ego_cav, timestamp_key, ego_cav_id, scenario_index):
        """
        Calculate the distance to ego for each cav.
        """
        # Find ego pose first
        ego_cav_content = scenario_data_on_ego_cav[ego_cav_id]

        if timestamp_key not in ego_cav_content:
            print(f"Warning: timestamp_key {timestamp_key} not in ego_cav_content for {self.idx2scenrio[scenario_index]}")
        ego_lidar_pose = load_yaml(ego_cav_content[timestamp_key]['yaml'])['lidar_pose']

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_data_on_ego_cav.items():
            cur_lidar_pose = \
                load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            # distance = \
            #     math.sqrt((cur_lidar_pose[0] -
            #                ego_lidar_pose[0]) ** 2 +
            #               (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            distance = dist_two_pose(cur_lidar_pose, ego_lidar_pose)
            cav_content['distance_to_ego'] = distance
            scenario_data_on_ego_cav.update({cav_id: cav_content})

        return ego_cav_content

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000  # in ms
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        # the data is 10 hz for both opv2v and v2v4real
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100  # number of frames
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask,
                flip=None, rotation=None, scale=None):
        """
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'flip': flip,
                    'noise_rotation': rotation,
                    'noise_scale': scale}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=self.cur_ego_pose_flag)

        processed_data_dict = OrderedDict()

        ego_cav_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_cav_id = int(cav_id)
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_scenario = cav_content['scenario_name']
                ego_timestamp_key = cav_content['timestamp_key']
                ego_timestamp_idx = cav_content['timestamp_idx']
                break

        assert ego_cav_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():

            transformation_matrix = \
                selected_cav_base['params']['transformation_matrix']
            # this is used to project gt objects to ego space
            gt_transformation_matrix = \
                selected_cav_base['params']['gt_transformation_matrix']

            selected_cav_processed = \
                self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix':
                                               transformation_matrix})
            selected_cav_processed.update({'gt_transformation_matrix':
                                               gt_transformation_matrix})

            # if int(cav_id) == ego_cav_id:
            #     processed_data_dict.update({'ego': selected_cav_processed})

            update_cav = "ego" if int(cav_id) == ego_cav_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        processed_data_dict['ego'].update(
            {
             "ego_lidar_pose": ego_lidar_pose,
             "ego_scenario": ego_scenario,
             "ego_timestamp_idx" : ego_timestamp_idx,
             "ego_timestamp_key": ego_timestamp_key,
             "ego_cav_id": ego_cav_id
            })
            
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)

        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       np.identity(4))
        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})

        return selected_cav_processed
        
    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        scenario_list = []
        timestamp_key_list = []
        timestamp_idx_list = []
        ego_cav_id_list = []
        lidar_pose_list = []

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]

                projected_lidar = cav_content['origin_lidar']
                projected_lidar[:, :3] = \
                    box_utils.project_points_by_matrix_torch(
                        projected_lidar[:, :3],
                        transformation_matrix)
                projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix'])).float()
            gt_transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['gt_transformation_matrix'])).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'gt_transformation_matrix': gt_transformation_matrix_torch})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

            if cav_id == 'ego':
                lidar_pose_list.append(cav_content['ego_lidar_pose'])
                ego_cav_id_list.append(cav_content['ego_cav_id'])
                timestamp_key_list.append(cav_content['ego_timestamp_key'])
                timestamp_idx_list.append(cav_content['ego_timestamp_idx'])
                scenario_list.append(cav_content['ego_scenario'])

        lidar_pose = torch.from_numpy(np.array(lidar_pose_list))

        if self.visualize:
            projected_lidar_stack = [torch.from_numpy(
                np.vstack(projected_lidar_list))]
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        output_dict['ego'].update({
            'lidar_pose': lidar_pose,
            'scenario_list': scenario_list,
            'timestamp_key_list': timestamp_key_list,
            'timestamp_idx_list': timestamp_idx_list,
            'ego_cav_id_list': ego_cav_id_list
        })
            
        return output_dict

    def post_process(self, data_dict, output_dict, for_tracking=False):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """

        if for_tracking:
            pred_box_tensor, pred_boxes3d, pred_score = \
                self.post_processor.post_process(data_dict, output_dict, for_tracking)
            gt_box_tensor, gt_object_id_tensor = self.post_processor.generate_gt_bbx(data_dict, for_tracking)

            return pred_box_tensor, pred_boxes3d, pred_score, gt_box_tensor, gt_object_id_tensor, output_dict
        else:
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor