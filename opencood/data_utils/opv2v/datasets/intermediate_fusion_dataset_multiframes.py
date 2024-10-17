"""
Dataset class for intermediate fusion
"""
import copy
import math
import os
import pickle
import random
import warnings
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.opv2v.datasets
import opencood.data_utils.opv2v.post_processor as post_processor
import opencood.utils.pcd_utils as pcd_utils
from MTR.mtr.datasets.opv2v_multiego_dataset import OPV2VMultiEgoDataset
from opencood.data_utils.opv2v.datasets import basedataset
from opencood.data_utils.opv2v.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.pcd_utils import (downsample_lidar_minimum,
                                      mask_ego_points, mask_points_by_range,
                                      shuffle_points)
from opencood.utils.transformation_utils import x1_to_x2


class IntermediateFusionDatasetMultiFrame(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDatasetMultiFrame, self). \
            __init__(params, visualize, train)

        self.past_frames = params['train_params']['past_frames']
        self.future_frames = params['train_params']['future_frames']
        self.queue_length = params['train_params']['queue_length']
        self.preprocessed_data = params['preprocessed_dir']
        self.ego_shift = params['train_params']['ego_shift']
        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']

        additional_root = params['additional_dir']
        lane_path = os.path.join(additional_root, 'train' if self.train else 'test')

        if 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']
        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x)) and x != '2021_09_09_13_20_58']) # 2021_09_09_13_20_58 doesn't have bev lane images
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []
        self.scenario_record = []
        self.cav_id = []

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

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)
                    lane_file = os.path.join(lane_path, scenario_folder.split('/')[-1], cav_id, f"{timestamp}_bev_lane.png")

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['lane'] = \
                        lane_file
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if self.train:
                    if j == 0:
                        # we regard the agent with the minimum id as the ego
                        self.scenario_database[i][cav_id]['ego'] = True
                    else:
                        self.scenario_database[i][cav_id]['ego'] = False

                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                    self.scenario_record.append(i)
                    self.cav_id.append(cav_id)
                else:
                    if j == 0:
                        # we regard the agent with the minimum id as the ego
                        self.scenario_database[i][cav_id]['ego'] = True
                        if not self.len_record:
                            self.len_record.append(len(timestamps))
                        else:
                            prev_last = self.len_record[-1]
                            self.len_record.append(prev_last + len(timestamps))
                        self.scenario_record.append(i)
                        self.cav_id.append(cav_id)
                    else:
                        self.scenario_database[i][cav_id]['ego'] = False

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.preprocessed_dir = params['preprocessed_dir']
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

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = \
                load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = \
                math.sqrt((cur_lidar_pose[0] -
                           ego_lidar_pose[0]) ** 2 +
                          (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            if 'distance_to_ego' not in cav_content:
                cav_content['distance_to_ego'] = [distance]
            else:
                cav_content['distance_to_ego'].append(distance)
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

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
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = self.scenario_record[i]
                cav_id = self.cav_id[i]
                break
        scenario_database = copy.deepcopy(self.scenario_database[scenario_index])

        if self.ego_shift:
            scenario_database[cav_id]['ego'] = True
            for rest_cav_id in list(scenario_database.keys()):
                if rest_cav_id == cav_id:
                    continue
                scenario_database[rest_cav_id]['ego'] = False

        timespan = self.future_frames + self.past_frames + 3
        if len(os.listdir(self.preprocessed_data)) != 0:
            while(True):
                scenario_name = self.idx2scenrio[scenario_index].split('/')[-1]
                preprocessed_file = os.path.join(self.preprocessed_data, 'train' if self.train else 'test', f"{scenario_name}-{cav_id}-traj.pickle")
                with open(preprocessed_file, 'rb') as f:
                    ret = pickle.load(f)
                    data, timestamps = ret['data'], ret['timestamps']
                if len(timestamps) < timespan:
                    scenario_index += 1
                    scenario_database = self.scenario_database[scenario_index]
                    cav_ids = list(scenario_database.keys())
                    cav_idx = random.randint(0, len(cav_ids)-1)
                    cav_id = cav_ids[cav_idx]
                else:
                    break
        else:
            data, timestamps = OPV2VMultiEgoDataset.get_gt_traj(scenario_database, cav_id)

        sdc_track_index = 0 #useless information for OPV2V
        current_time_index = random.randint(self.past_frames, len(timestamps)-(self.future_frames+3)) # make sure 1 seconds before and 5 second after
        obj_trajs_full = np.array(list(data.values()))
        obj_trajs_full[:, :, 6:7] = np.radians(obj_trajs_full[:, :, 6:7])
        while(sum(obj_trajs_full[:, current_time_index, -1]) == 0):
            current_time_index = random.randint(self.past_frames, len(timestamps)-(self.future_frames+3))
        cur_timestamp = timestamps[current_time_index]
        timestamps = np.array([0.1 * (0.1 * i) for i in range(len(timestamps))], dtype=float) # convert to 0.1s incremented timestamps
        timestamps = timestamps[current_time_index-self.past_frames:current_time_index + 1] # past to current frame

        track_index_to_predict = np.array(list(range(len(data))))
        obj_types = np.array(['TYPE_VEHICLE']*len(data))
        obj_ids = np.array([key for key in data.keys()])
        
        obj_trajs_past = obj_trajs_full[:, current_time_index-self.past_frames:current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:current_time_index + 1+self.future_frames]

        center_objects, track_index_to_predict = OPV2VMultiEgoDataset.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scenario_index
        )

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = OPV2VMultiEgoDataset.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
        )

        traj_data = {
            'obj_trajs_data': obj_trajs_pos,
            'obj_trajs_mask': obj_trajs_mask,
            'obj_trajs_future_state': obj_trajs_future_state, 
            'obj_trajs_future_mask': obj_trajs_future_mask
        }

        # check the timestamp index
        timestamp_index = current_time_index
        timestamp_indexs = list(range(timestamp_index - self.queue_length+1, timestamp_index+1))
        # retrieve the corresponding timestamp key
        # timestamp_key = self.return_timestamp_key(scenario_database,
        #                                           timestamp_index)
        timestamp_keys = []
        for timestamp_i in range(timestamp_index - self.queue_length+1, timestamp_index+1):
            timestamp_keys.append(self.return_timestamp_key(scenario_database, timestamp_i))
            
        # calculate distance to ego for each cav
        ego_cav_contents = []
        for timestamp_key in timestamp_keys:
            ego_cav_contents.append(self.calc_dist_to_ego(scenario_database, timestamp_key))

        data = OrderedDict()
        # load files for all CAVs

        for j, timestamp_index, ego_cav_content in zip(range(self.queue_length), timestamp_indexs, ego_cav_contents):
            timestamp_key = timestamp_keys[j]
            data[timestamp_key] = OrderedDict()
            for cav_id, cav_content in scenario_database.items():
                data[timestamp_key][cav_id] = OrderedDict()
                data[timestamp_key][cav_id]['ego'] = cav_content['ego']

                # calculate delay for this vehicle
                timestamp_delay = \
                    self.time_delay_calculation(cav_content['ego'])

                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index
                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
                timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                                timestamp_index_delay)
                # add time delay to vehicle parameters
                data[timestamp_key][cav_id]['time_delay'] = timestamp_delay
                # load the corresponding data into the dictionary
                data[timestamp_key][cav_id]['params'] = self.reform_param(cav_content,
                                                        ego_cav_content,
                                                        timestamp_key,
                                                        timestamp_key_delay,
                                                        cur_ego_pose_flag)
                data[timestamp_key][cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
                data[timestamp_key][cav_id]['scenario'] = scenario_index
                data[timestamp_key][cav_id]['timestamp'] = timestamp_key

        return data, traj_data

    def __getitem__(self, idx):
        base_data_dict, traj_data = self.retrieve_base_data(idx,
                                                 cur_ego_pose_flag=self.cur_ego_pose_flag)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_poses = OrderedDict()
        ego_scenarios = OrderedDict()
        ego_timestamps = OrderedDict()
        # first find the ego vehicle's lidar pose
        for timestamp, contents in base_data_dict.items():
            for cav_id, cav_content in contents.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_scenario = cav_content['scenario']
                    ego_timestamp = cav_content['timestamp']
                    ego_lidar_poses[timestamp] = ego_lidar_pose
                    ego_scenarios[timestamp] = ego_scenario
                    ego_timestamps[timestamp] = ego_timestamp
                    break
            # else:
            #     continue
            # break
        # assert cav_id == list(base_data_dict.keys())[
        #     0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        processed_features = OrderedDict()
        object_stack = OrderedDict()
        object_id_stack = OrderedDict()

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocities = OrderedDict()
        time_delays = OrderedDict()
        infras = OrderedDict()
        spatial_correction_matrices = OrderedDict()

        if self.visualize:
            projected_lidar_stack = OrderedDict()

        # loop over all CAVs to process information
        for timestamp, contents in base_data_dict.items():
            processed_features[timestamp] = []
            object_stack[timestamp] = []
            object_id_stack[timestamp] = []
            velocities[timestamp] = []
            time_delays[timestamp] = []
            infras[timestamp] = []
            spatial_correction_matrices[timestamp] = []
            ego_lidar_pose = ego_lidar_poses[timestamp]
            if self.visualize:
                projected_lidar_stack[timestamp] = []
            for cav_id, selected_cav_base in contents.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > opencood.data_utils.opv2v.datasets.COM_RANGE:
                    continue

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_lidar_pose)

                object_stack[timestamp].append(selected_cav_processed['object_bbx_center'])
                object_id_stack[timestamp] += selected_cav_processed['object_ids']
                processed_features[timestamp].append(
                    selected_cav_processed['processed_features'])

                velocities[timestamp].append(selected_cav_processed['velocity'])
                time_delays[timestamp].append(float(selected_cav_base['time_delay']))
                # this is only useful when proj_first = True, and communication
                # delay is considered. Right now only V2X-ViT utilizes the
                # spatial_correction. There is a time delay when the cavs project
                # their lidar to ego and when the ego receives the feature, and
                # this variable is used to correct such pose difference (ego_t-1 to
                # ego_t)
                spatial_correction_matrices[timestamp].append(
                    selected_cav_base['params']['spatial_correction_matrix'])
                infras[timestamp].append(1 if int(cav_id) < 0 else 0)

                if self.visualize:
                    projected_lidar_stack[timestamp] = selected_cav_processed['projected_lidar']

        
        object_bbx_centers = OrderedDict()
        object_bbx_masks = OrderedDict()
        object_ids = OrderedDict()
        anchor_boxes = OrderedDict()
        processed_lidars = OrderedDict()
        label_dicts = OrderedDict()
        processed_velocities = OrderedDict()
        processed_time_delays = OrderedDict()
        processed_infras = OrderedDict()
        processed_spatial_correction_matrices = OrderedDict()


        for timestamp, object_id, object, processed_feature, velocity, time_delay, infra, spatial_correction_matrix in \
            zip(object_id_stack.keys(), object_id_stack.values(), object_stack.values(), processed_features.values(), velocities.values(), time_delays.values(), infras.values(), spatial_correction_matrices.values()):
            # exclude all repetitive objects

            unique_indices = \
                [object_id.index(x) for x in set(object_id)]
            object = np.vstack(object)
            object = object[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object.shape[0], :] = object
            mask[:object.shape[0]] = 1

            # merge preprocessed features from different cavs into the same dict
            cav_num = len(processed_feature)
            merged_feature_dict = self.merge_features_to_dict(processed_feature)

            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            label_dict = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)

            # pad dv, dt, infra to max_cav
            velocity = velocity + (self.max_cav - len(velocity)) * [0.]
            time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
            infra = infra + (self.max_cav - len(infra)) * [0.]
            spatial_correction_matrix = np.stack(spatial_correction_matrix)
            padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                                spatial_correction_matrix),1,1))
            spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                    padding_eye], axis=0)

            object_bbx_centers[timestamp] = object_bbx_center
            object_bbx_masks[timestamp] = mask
            object_ids[timestamp] = [object_id[i] for i in unique_indices]
            anchor_boxes[timestamp] = anchor_box
            processed_lidars[timestamp] = merged_feature_dict
            label_dicts[timestamp] = label_dict
            processed_velocities[timestamp] = velocity
            processed_time_delays[timestamp] = time_delay
            processed_infras[timestamp] = infra
            processed_spatial_correction_matrices[timestamp] = spatial_correction_matrix

        obj_trajs_data = traj_data['obj_trajs_data']
        obj_trajs_mask = traj_data['obj_trajs_mask']
        obj_trajs_future_state = traj_data['obj_trajs_future_state']
        obj_trajs_future_mask = traj_data['obj_trajs_future_mask']

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_centers,
             'object_bbx_mask': object_bbx_masks,
             'object_ids': object_ids,
             'anchor_box': anchor_boxes,
             'processed_lidar': processed_lidars,
             'label_dict': label_dicts,
             'cav_num': cav_num,
             'velocity': processed_velocities,
             'time_delay': processed_time_delays,
             'infra': processed_infras,
             'spatial_correction_matrix': processed_spatial_correction_matrices,
             'pairwise_t_matrix': pairwise_t_matrix,
             "ego_lidar_pose": ego_lidar_poses,
             "ego_scenario": ego_scenarios,
             "ego_timestamp": ego_timestamps,
             'obj_trajs_data': obj_trajs_data,
             'obj_trajs_mask': obj_trajs_mask,
             'obj_trajs_future_state': obj_trajs_future_state,
             'obj_trajs_future_mask': obj_trajs_future_mask})

        if self.visualize:
            # processed_data_dict['ego'].update({'origin_lidar':
            #     np.vstack(
            #         projected_lidar_stack)}) #?
            processed_data_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
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

        # calculate the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

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
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center_full = []
        object_bbx_mask_full = []
        object_ids_full = []
        processed_lidar_list_full = []
        # used to record different scenario
        record_len_full = []
        label_dict_list_full = []
        lidar_pose_list_full = []
        scenario_list_full = []
        timestamp_list_full = []
        prior_encodings = []

        # pairwise transformation matrix
        pairwise_t_matrix_list_full = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list_full = []

        for t in range(self.queue_length):
            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            scenario_list = []
            timestamp_list = []

            # used for PriorEncoding for models
            velocity = []
            time_delay = []
            infra = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # used for correcting the spatial transformation between delayed timestamp
            # and current timestamp
            spatial_correction_matrix_list = []

            if self.visualize:
                origin_lidar = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                timestamps = list(ego_dict['object_bbx_center'].keys())
                cur_timestamp = timestamps[t]
            
                object_bbx_center.append(ego_dict['object_bbx_center'][cur_timestamp])
                object_bbx_mask.append(ego_dict['object_bbx_mask'][cur_timestamp])
                object_ids.append(ego_dict['object_ids'][cur_timestamp])

                processed_lidar_list.append(ego_dict['processed_lidar'][cur_timestamp])
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'][cur_timestamp])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                velocity.append(ego_dict['velocity'][cur_timestamp])
                time_delay.append(ego_dict['time_delay'][cur_timestamp])
                infra.append(ego_dict['infra'][cur_timestamp])
                spatial_correction_matrix_list.append(
                    ego_dict['spatial_correction_matrix'][cur_timestamp])
                lidar_pose_list.append(ego_dict['ego_lidar_pose'][cur_timestamp])
                scenario_list.append(ego_dict['ego_scenario'][cur_timestamp])
                timestamp_list.append(ego_dict['ego_timestamp'][cur_timestamp])
                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])
            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # example: {'voxel_features':[np.array([1,2,3]]),
            # np.array([3,5,6]), ...]}
            merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(merged_feature_dict)
            # [2, 3, 4, ..., M], M <= max_cav
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # (B, max_cav)
            velocity = torch.from_numpy(np.array(velocity))
            time_delay = torch.from_numpy(np.array(time_delay))
            infra = torch.from_numpy(np.array(infra))
            spatial_correction_matrix_list = \
                torch.from_numpy(np.array(spatial_correction_matrix_list))
            # (B, max_cav, 3)
            prior_encoding = \
                torch.stack([velocity, time_delay, infra], dim=-1).float()
            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # (B, 6)
            lidar_pose = torch.from_numpy(np.array(lidar_pose_list))

            object_bbx_center_full.append(object_bbx_center)
            object_bbx_mask_full.append(object_bbx_mask)
            processed_lidar_list_full.append(processed_lidar_torch_dict)
            record_len_full.append(record_len)
            label_dict_list_full.append(label_torch_dict)
            object_ids_full.append(object_ids[0])
            prior_encodings.append(prior_encoding)
            spatial_correction_matrix_list_full.append(spatial_correction_matrix_list)
            pairwise_t_matrix_list_full.append(pairwise_t_matrix)
            lidar_pose_list_full.append(lidar_pose)
            scenario_list_full.append(scenario_list)
            timestamp_list_full.append(timestamp_list)

        object_bbx_center = torch.stack(object_bbx_center_full, dim=0)
        object_bbx_mask = torch.stack(object_bbx_mask_full, dim=0)
        # processed_lidar_torch_dict = torch.stack(processed_lidar_list_full, dim=0)
        prior_encoding = torch.stack(prior_encodings, dim=0)
        spatial_correction_matrix_list = torch.stack(spatial_correction_matrix_list_full, dim=0)
        lidar_pose = torch.stack(lidar_pose_list_full, dim=0)

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_list_full,
                                   'record_len': record_len, #?
                                   'label_dict': label_dict_list_full,
                                   'object_ids': object_ids_full,
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'pairwise_t_matrix': pairwise_t_matrix, #bs, frames, cav, cav, 4, 4
                                   'lidar_pose': lidar_pose,
                                   'scenario_list': scenario_list_full,
                                   'timestamp_list': timestamp_list_full})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                {timestamp: torch.from_numpy(batch[0]['ego']['anchor_box'][timestamp]) for timestamp in batch[0]['ego']['anchor_box'].keys()}}
            )

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
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
        pred_box_tensor_list, pred_boxes3d_list, pred_score_list = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor_list = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor_list, pred_boxes3d_list, pred_score_list, gt_box_tensor_list

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        timestamps = list(base_data_dict.keys())
        timespan = len(timestamps)
        pairwise_t_matrix = np.zeros((timespan, max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :, :] = np.identity(4)
        else:
            # save all transformation matrix in a list in order first.
            for timeindex, contents in zip(range(timespan), base_data_dict.values()):
                t_list = []
                for cav_id, cav_content in contents.items():
                    t_list.append(cav_content['params']['transformation_matrix'])

                for i in range(len(t_list)):
                    for j in range(len(t_list)):
                        # identity matrix to self
                        if i == j:
                            t_matrix = np.eye(4)
                            pairwise_t_matrix[timeindex, i, j] = t_matrix
                            continue
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        pairwise_t_matrix[timeindex, i, j] = t_matrix

        return pairwise_t_matrix
