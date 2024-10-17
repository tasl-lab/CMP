import os
import sys

import numpy as np
from scipy.spatial import ConvexHull
import pickle
import torch
from PIL import Image
import copy
from tqdm import tqdm

from mtr.datasets.dataset import DatasetTemplate
from mtr.utils import common_utils

import math
from collections import OrderedDict, Counter


from opencood.hypes_yaml.yaml_utils import load_yaml

class OPV2VMultiEgoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, logger=None, visualize=False):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        self.is_train = training
        self.idx2scenrio = {}

        if self.is_train:
            root_dir = dataset_cfg['train_dir']
        else:
            root_dir = dataset_cfg['validate_dir']

        lane_path = os.path.join(dataset_cfg['additional_dir'], 'train' if self.is_train else 'test')

        self.max_cav = 7 if 'max_cav' not in dataset_cfg else dataset_cfg['max_cav']

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir,
                                                              x)) and x != '2021_09_09_13_20_58'])  # 2021_09_09_13_20_58 doesn't have bev lane images
    
        # Structure: {scenario_id :
        #                       {cav_1 :
        #                           {timestamp1 :
        #                            {yaml: path,
        #                             lidar: path,
        #                             cameras:list of path,
        #                             lane: path}
        #                           }
        #                       }
        #             }
        self.scenario_database = OrderedDict()  # save all scenarios
        self.records = []

        dataset_cache_file_path = (self.dataset_cfg.DATASET_CACHE_DIR + '/opv2v_multiego_' +
                                   self.dataset_cfg.preprocessed_gt_traj_dir.split('/')[-1] + '_' +
                                   self.dataset_cfg.preprocessed_pred_traj_dir.split('/')[-1] + '_' +
                                   ('train' if self.is_train else 'test') +
                                   ('_' + str(self.dataset_cfg.CONVEX_HULL_THRESHOLD) if self.dataset_cfg.CONVEX_HULL_THRESHOLD != -1 else '') + '_dataset_cache.pkl')
        if os.path.exists(dataset_cache_file_path):
            logger.info('Loading cached dataset records. To reload, delete ' + dataset_cache_file_path)
            with open(dataset_cache_file_path, 'rb') as f:
                ret = pickle.load(f)
                self.scenario_database, self.records, self.idx2scenrio = ret['scenario_database'], ret['records'], ret[
                    'idx2scenrio']
            logger.info(f'Loaded {len(self.records)} records for {len(self.scenario_database)} scenarios.')
            return

        # loop over all scenarios
        logger.info('Collecting file paths...')
        for (i, scenario_folder) in tqdm(enumerate(scenario_folders)):
            self.scenario_database.update({i: OrderedDict()})
            self.idx2scenrio[i] = scenario_folder
            scenario_name = scenario_folder.split('/')[-1]

            # if scenario_name == '2021_09_09_13_20_58':
            #     continue  # No BEV lane image

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

            # Some CAVs have longer timestamps than others. We first need to find the min timestamp.
            minimum_set_of_timestamps = None
            for cav_id in cav_list:
                cav_path = os.path.join(scenario_folder, cav_id)
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)
                minimum_set_of_timestamps = timestamps if minimum_set_of_timestamps is None else \
                    (minimum_set_of_timestamps if len(minimum_set_of_timestamps) < len(timestamps) else timestamps)

            # Skip insufficient lengthed trajectories.
            minimum_timespan = self.dataset_cfg.FUTURE_FRAMES + self.dataset_cfg.PAST_FRAMES + 3
            if len(minimum_set_of_timestamps) < minimum_timespan:
                continue

            # loop over all CAV data in a scenario
            for cav_id in cav_list:
                ## Uncomment below to limit # neighbor AI to 0, essentially disable comms.
                # if cav_id != ego_cav_id:
                #     continue

                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                for (timestamp_idx, timestamp_key) in enumerate(minimum_set_of_timestamps):
                    self.scenario_database[i][cav_id][timestamp_key] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path, timestamp_key + '.yaml')
                    lidar_file = os.path.join(cav_path, timestamp_key + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp_key)
                    lane_file = os.path.join(lane_path, scenario_folder.split('/')[-1], cav_id,
                                             f"{timestamp_key}_bev_lane.png")
                    fused_feature_file = os.path.join(self.dataset_cfg.cobevt_fused_features_dir,
                                                      f"fused_feature_{scenario_folder.split('/')[-1]}_{cav_id}_{timestamp_idx}.pkl.npy")
                    preprocessed_gt_file_path = os.path.join(self.dataset_cfg.preprocessed_gt_traj_dir,
                                                             'train' if self.is_train else 'test',
                                                             f"{scenario_name}-{cav_id}-traj.pickle")
                    preprocessed_pred_file_path = os.path.join(self.dataset_cfg.preprocessed_pred_traj_dir,
                                                               'train' if self.is_train else 'test',
                                                               f"{scenario_name}-{cav_id}-traj.pickle")

                    self.scenario_database[i][cav_id][timestamp_key]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp_key]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp_key]['camera0'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp_key]['lane'] = \
                        lane_file
                    self.scenario_database[i][cav_id][timestamp_key]['fused_feature'] = \
                        fused_feature_file
                    self.scenario_database[i][cav_id][timestamp_key]['preprocessed_gt_data'] = \
                        preprocessed_gt_file_path
                    self.scenario_database[i][cav_id][timestamp_key]['preprocessed_pred_data'] = \
                        preprocessed_pred_file_path

        logger.info('Validating dataset.')
        cavs_covex_covered_areas = []
        for scenario_idx, scenario_content in tqdm(self.scenario_database.items()):
            for ego_cav_id, ego_cav_content in scenario_content.items():
                for timestamp_idx, (timestamp_key, timestamp_content) in enumerate(ego_cav_content.items()):

                    # Make sure the corresponding trajectories contain valid segments.
                    cav_ids_that_have_valid_data = []
                    at_least_one_cav_has_valid_segments = False
                    for cav_id in scenario_content.keys():
                        if not os.path.exists(scenario_content[cav_id][timestamp_key]['preprocessed_gt_data']) or \
                                not os.path.exists(scenario_content[cav_id][timestamp_key]['preprocessed_pred_data']):
                            continue

                        with open(scenario_content[cav_id][timestamp_key]['preprocessed_gt_data'], 'rb') as f:
                            ret = pickle.load(f)
                            gt_data, timestamp_keys_gt = ret['data'], ret['timestamps']
                        with open(scenario_content[cav_id][timestamp_key]['preprocessed_pred_data'], 'rb') as f:
                            ret = pickle.load(f)
                            pred_data, timestamp_keys_pred = ret['data'], ret['timestamps']
                        gt_data, pred_data = self.extract_common_key_value_pairs_and_sort(gt_data, pred_data)
                        valid_pred_data, all_valid_pred_data, valid_gt_data, all_valid_gt_data = self.extract_frames(
                            gt_data, pred_data, timestamp_idx)
                        if len(valid_pred_data) != 0 and len(valid_gt_data) != 0:
                            at_least_one_cav_has_valid_segments = True
                            cav_ids_that_have_valid_data.append(cav_id)
                        # elif timestamp_idx > 10 and timestamp_idx < 128:
                        #     logger.warning(f'{self.idx2scenrio[scenario_idx].split("/")[-1]}-{ego_cav_id}-{timestamp_idx} has no valid segment')
                        #     break
                    if not at_least_one_cav_has_valid_segments:
                        continue

                    # Make sure the fused feature exists.
                    if not os.path.exists(timestamp_content['fused_feature']):
                        logger.warning(f'{self.idx2scenrio[scenario_idx].split("/")[-1]}-{ego_cav_id}-{timestamp_idx} has no fused feature')
                        continue

                    # Make sure all CAVs in the scene have bev lane.
                    all_cavs_this_moment_have_bev_lane = True
                    for cav_id in cav_ids_that_have_valid_data:
                        if not os.path.exists(scenario_content[cav_id][timestamp_key]['lane']):
                            all_cavs_this_moment_have_bev_lane = False
                            break
                    if not all_cavs_this_moment_have_bev_lane:
                        logger.warning(f'{self.idx2scenrio[scenario_idx].split("/")[-1]}-{ego_cav_id}-{timestamp_idx} {cav_id} has no bev lane')
                        continue
                    
                    # Convex Hull Filter.
                    covex_hull_area = -1
                    if self.dataset_cfg.CONVEX_HULL_THRESHOLD != -1:
                        cav_starting_positions = []

                        if len(scenario_content.keys()) <= 2:
                            # 1 or 2 CAVs cannot form a 2D shape.
                            covex_hull_area = 0.0
                        else:
                            # Gather the starting position of the all CAVs.
                            for cav_id in scenario_content.keys():
                                cur_lidar_pose = load_yaml(scenario_content[cav_id][timestamp_key]['yaml'])['lidar_pose']
                                cav_starting_positions.append(cur_lidar_pose[:2])

                            # Compute the area they cover.
                            hull = ConvexHull(cav_starting_positions)
                            covex_hull_area = hull.area

                        cavs_covex_covered_areas.append(covex_hull_area)  # Debug Statistics

                        # Skip if the area is below our threashold.
                        if covex_hull_area < self.dataset_cfg.CONVEX_HULL_THRESHOLD:
                            continue

                    assert covex_hull_area >= self.dataset_cfg.CONVEX_HULL_THRESHOLD

                    # Record a single training instance.
                    self.records.append({
                        'scenario_idx': scenario_idx,
                        'ego_cav_id': ego_cav_id,
                        'timestamp_key': timestamp_key,
                        'timestamp_idx': timestamp_idx,
                        'cav_ids_that_have_valid_data': cav_ids_that_have_valid_data,
                        'covex_hull_area': covex_hull_area
                    })

        # Plot the distribution of CAV convex hull areas.
        # plt.hist(cavs_covex_covered_areas, bins=10, edgecolor='black')
        # plt.xlabel('Area')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Convex Hull Areas Formed by >2 CAVs')
        # plt.savefig('./Visualization/dist.png')
        # plt.close()     
        # exit(0)

        # Convert scenario_database to be ordered by scenario, who_is_ego, timestamp.
        unique_records = {}
        for record in self.records:
            key = (record['scenario_idx'], record['ego_cav_id'], record['timestamp_key'])
            if key not in unique_records:
                unique_records[key] = record

        # Extract values and sort
        deduplicated_sorted_records = sorted(unique_records.values(), key=lambda x: (
            x['scenario_idx'], record['ego_cav_id'], x['timestamp_key']))
        self.records = deduplicated_sorted_records

        with open(dataset_cache_file_path, 'wb') as f:
            pickle.dump(
                {'scenario_database': self.scenario_database, 'records': self.records, 'idx2scenrio': self.idx2scenrio},
                f)

        logger.info(f'Preparing {len(self.records)} records for {len(self.scenario_database)} scenarios')
        list_of_num_cavs_per_scene_count = [len(x['cav_ids_that_have_valid_data']) for x in self.records]
        logger.info(f'The distribution of num_cavs in the scenes is: {Counter(list_of_num_cavs_per_scene_count)}')


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

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

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
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})

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
        if self.dataset_cfg.ASYNC == 'real':
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.dataset_cfg.ASYNC_OVERHEAD)
            tc = self.dataset_cfg.DATA_SIZE / self.dataset_cfg.TRANSMISSION_SPEED * 1000
            time_delay = int(overhead_noise + tc + self.dataset_cfg.BACKBONE_DELAY)
        elif self.dataset_cfg.ASYNC == 'sim':
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.dataset_cfg.ASYNC_OVERHEAD)

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.dataset_cfg.ASYNC is not False else 0

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

    @staticmethod
    def calculate_velocity_components(v, roll, pitch, yaw):
        # Convert angles from degrees to radians
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        v_x = v * math.cos(yaw) * math.cos(pitch)
        v_y = v * math.sin(yaw) * math.cos(pitch)
        v_z = v * math.sin(pitch)

        return v_x, v_y, v_z

    @staticmethod
    def interpolate_car_info(cars_data):

        for car_id, car_states in cars_data.items():
            # get all hop

            valid_indices = []
            for i in range(len(car_states) - 1):
                if car_states[i][-1] ^ car_states[i + 1][-1] == 1:
                    if car_states[i][-1] == 0:
                        valid_indices.append(i + 1)
                    elif car_states[i][-1] == 1:
                        valid_indices.append(i)

            valid_indices = sorted(list(set(valid_indices)))  # remove duplicate items
            for pre, post in zip(valid_indices[:-1], valid_indices[1:]):
                missing_frames = post - pre - 1
                if car_states[pre + 1][-1] == 1:  # no need for padding
                    continue
                missing_states = OPV2VMultiEgoDataset.interpolate_missing_frames(car_states[pre], car_states[post],
                                                                                 missing_frames)
                for j, state in zip(range(pre, post), missing_states):
                    cars_data[car_id][j + 1] = copy.deepcopy(state)

        return cars_data

    @staticmethod
    def interpolate_missing_frames(prev_state, curr_state, missing_frames):
        missing_states = []
        for i in range(1, missing_frames + 1):
            weight_prev = (missing_frames - i + 1) / (missing_frames + 1)
            weight_curr = (i) / (missing_frames + 1)
            interpolated_state = OPV2VMultiEgoDataset.interpolate_states(prev_state, curr_state, weight_prev,
                                                                         weight_curr)
            missing_states.append(interpolated_state)
        return missing_states

    @staticmethod
    def interpolate_states(prev_state, curr_state, weight_prev, weight_curr):

        x = OPV2VMultiEgoDataset.interpolate_value(prev_state[0], curr_state[0], weight_prev, weight_curr)
        y = OPV2VMultiEgoDataset.interpolate_value(prev_state[1], curr_state[1], weight_prev, weight_curr)
        z = prev_state[2]
        l = prev_state[3]
        w = prev_state[4]
        h = prev_state[5]
        heading = OPV2VMultiEgoDataset.interpolate_value(prev_state[6], curr_state[6], weight_prev, weight_curr)
        # v_x = OPV2VMultiEgoDataset.interpolate_value(prev_state[7], curr_state[7], weight_prev, weight_curr)
        # v_y = OPV2VMultiEgoDataset.interpolate_value(prev_state[8], curr_state[8], weight_prev, weight_curr)
        # confidence = prev_state[-2]
        valid = 1

        return [x, y, z, l, w, h, heading, valid]

    @staticmethod
    def interpolate_value(prev_value, curr_value, weight_prev, weight_curr):
        interpolated_value = (prev_value * weight_prev) + (curr_value * weight_curr)
        return interpolated_value

    @staticmethod
    def get_gt_traj(scenario_instance):
        """
            Note here we do not extract the ego traj.
        """

        cur_cav_content = scenario_instance

        timestamps = [key for key in cur_cav_content.keys() if key.startswith('0')]
        data = OrderedDict()

        # loop over all timestamps
        for i, timestamp in enumerate(timestamps):
            car_manipulated = dict.fromkeys(data.keys(), 0)
            cur_params = load_yaml(cur_cav_content[timestamp]['yaml'])
            # ego_params = load_yaml(ego_cav_content[timestamp]['yaml'])
            # ego_lidar_pose = ego_params['lidar_pose']

            # iterate all neighbors.
            for vehicle_id, vehicle_params in cur_params['vehicles'].items():
                if vehicle_id not in data.keys():
                    data[vehicle_id] = []
                    surrounding_car = cur_params['vehicles'][vehicle_id]
                    for j in range(i):
                        data[vehicle_id].append(
                            [0, 0, 0, 0, 0, 0, 0, 0])  # padding invalid state for the beginning
                    roll, yaw, pitch = surrounding_car['angle']
                    # v_x, v_y, v_z = OPV2VMultiEgoDataset.calculate_velocity_components(surrounding_car['speed'], roll, pitch,
                    #                                                            yaw)  # normalize velocity by average speed 30km/h
                    xyz = [loc + cen for loc, cen in zip(surrounding_car['location'], surrounding_car['center'])]
                    data[vehicle_id].append([*xyz, *surrounding_car['extent'], surrounding_car['angle'][1],
                                             1])  # x, y, z, l, w, h, theta, valid
                else:
                    surrounding_car = cur_params['vehicles'][vehicle_id]
                    roll, yaw, pitch = surrounding_car['angle']
                    # v_x, v_y, v_z = OPV2VMultiEgoDataset.calculate_velocity_components(surrounding_car['speed'], roll, pitch,
                    #                                                            yaw)
                    xyz = [loc + cen for loc, cen in zip(surrounding_car['location'], surrounding_car['center'])]
                    data[vehicle_id].append([*xyz, *surrounding_car['extent'], surrounding_car['angle'][1],
                                             1])  # x, y, z, l, w, h, theta, valid

                car_manipulated[vehicle_id] = 1

            for vehicle_id, is_operated in car_manipulated.items():
                if is_operated == 0:
                    data[vehicle_id].append([0, 0, 0, 0, 0, 0, 0, 0])  # padding invalid state for the end
        # padding past
        for car_id, timestamp_content in data.items():
            idx = None
            for i in range(0, len(timestamps) - 2):
                if timestamp_content[i][-1] == 0 and timestamp_content[i + 1][-1] == 1:
                    idx = i
                    break
                elif timestamp_content[i][-1] == 0 and timestamp_content[i + 1][-1] == 0:
                    continue
                elif timestamp_content[i][-1] == 1:
                    break

            if idx is not None:
                for j in range(idx, -1, -1):
                    data[car_id][j] = copy.deepcopy(data[car_id][j + 1])
                    data[car_id][j][-1] = 0  # valid

        data = OPV2VMultiEgoDataset.interpolate_car_info(data)



        return data, timestamps

    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(
                np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(
                np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(
                np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        return self.retrieve_base_data(index)

    def extract_frames(self, gt_data, pred_data, cur_t):

        valid_gt_data = {}
        all_valid_gt_data = {}
        valid_pred_data = {}
        all_valid_pred_data = {}

        # Helper function to check if at least one frame is valid
        def are_all_frames_valid(frames):
            return all(frame[-1] == 1 for frame in frames)

        # Extract past and future frames for each car
        for car_id, pred_states in pred_data.items():
            # Ensure there are at least 10 past frames and 50 future frames available
            if cur_t >= self.dataset_cfg.PAST_FRAMES and cur_t + self.dataset_cfg.FUTURE_FRAMES < len(gt_data[car_id]):
                past_pred_frames = pred_states[cur_t - self.dataset_cfg.PAST_FRAMES:cur_t + 1]
                all_past_pred_frames = pred_states[:cur_t + 1]
                future_gt_frames = gt_data[car_id][cur_t + 1:cur_t + self.dataset_cfg.FUTURE_FRAMES + 1]
                all_future_pred_frames = gt_data[car_id][cur_t + 1:]

                # Check if at least one past and one future frame are valid
                if are_all_frames_valid(past_pred_frames) and are_all_frames_valid(future_gt_frames):
                    valid_gt_data[car_id] = future_gt_frames
                    all_valid_gt_data[car_id] = all_future_pred_frames
                    valid_pred_data[car_id] = past_pred_frames
                    all_valid_pred_data[car_id] = all_past_pred_frames

        return valid_pred_data, all_valid_pred_data, valid_gt_data, all_valid_gt_data

    def extract_common_key_value_pairs_and_sort(self, gt_data, pred_data):
        overlaped_cavs = gt_data.keys() & pred_data.keys()
        gt_data = {key: item for key, item in gt_data.items() if key in overlaped_cavs}
        pred_data = {key: item for key, item in pred_data.items() if key in overlaped_cavs}
        gt_data = OrderedDict(sorted(gt_data.items(), key=lambda x: x[0]))
        pred_data = OrderedDict(sorted(pred_data.items(), key=lambda x: x[0]))
        return gt_data, pred_data

    def organize_trajectory_timestamp_into_dict(self, gt_data, pred_data, timestamp_keys,
                                                scenario_index, scenario_name, cav_id, ego_cav_id,
                                                curr_timestamp_key, curr_timestamp_idx):
        # Find the intersection of the cav_ids.
        
        gt_data, pred_data = self.extract_common_key_value_pairs_and_sort(gt_data, pred_data)

        sdc_track_index = 0  # useless information for OPV2V

        gt_timestamp_seconds = np.array([0.1 * (0.1 * i) for i in range(len(timestamp_keys))],
                                        dtype=float)  # convert to 0.1s incremented timestamps
        cur_timestamp_seconds = gt_timestamp_seconds[
                                curr_timestamp_idx - self.dataset_cfg.PAST_FRAMES: curr_timestamp_idx + 1]

        obj_trajs_past, obj_trajs_past_full, obj_trajs_future, obj_trajs_future_full = self.extract_frames(
            gt_data, pred_data, curr_timestamp_idx)

        assert len(obj_trajs_past) != 0 and len(obj_trajs_future) != 0

        obj_types = np.array(['TYPE_VEHICLE'] * len(obj_trajs_past))
        obj_ids = np.array([key for key in obj_trajs_past.keys()])
        track_index_to_predict = np.array(list(range(len(obj_trajs_past))))
        obj_trajs_full = {key: item1 + item2 for key, item1, item2 in
                          zip(obj_trajs_past.keys(), obj_trajs_past_full.values(), obj_trajs_future_full.values())}

        # Convert list to ndarray.
        obj_trajs_past = np.array(list(obj_trajs_past.values()))
        obj_trajs_future = np.array(list(obj_trajs_future.values()))
        obj_trajs_full = np.array(list(obj_trajs_full.values()))

        # Convert to radians.
        obj_trajs_past[:, :, 6:7] = np.radians(obj_trajs_past[:, :, 6:7])
        obj_trajs_future[:, :, 6:7] = np.radians(obj_trajs_future[:, :, 6:7])
        obj_trajs_full[:, :, 6:7] = np.radians(obj_trajs_full[:, :, 6:7])

        cur_timestamp_seconds = gt_timestamp_seconds[
                                curr_timestamp_idx - self.dataset_cfg.PAST_FRAMES:curr_timestamp_idx + 1]  # past to current frame
        
        center_objects, track_index_to_predict = OPV2VMultiEgoDataset.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=curr_timestamp_idx,
            obj_types=obj_types, scene_id=scenario_index
        )

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
         obj_trajs_future_mask, center_gt_trajs,
         center_gt_trajs_mask, center_gt_final_valid_idx,
         track_index_to_predict_new, sdc_track_index_new, obj_types,
         obj_ids) = OPV2VMultiEgoDataset.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=cur_timestamp_seconds, obj_types=obj_types, obj_ids=obj_ids
        )

        ret_dict = {
            'scenario_id': np.array([scenario_index] * len(track_index_to_predict)),
            'scenario_name': np.array([scenario_name] * len(track_index_to_predict)),
            'ego_cav_id': np.array([ego_cav_id] * len(track_index_to_predict)),
            'timestamp_idx': np.array([curr_timestamp_idx] * len(track_index_to_predict)),
            'timestamp_key': np.array([curr_timestamp_key] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': obj_ids[track_index_to_predict_new],
            'center_objects_type': obj_types[track_index_to_predict_new],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict][:,
                                   curr_timestamp_idx - self.dataset_cfg.PAST_FRAMES:curr_timestamp_idx + 1 + self.dataset_cfg.FUTURE_FRAMES,
                                   :],
        }

        # Load BEV image.
        lane_path = self.scenario_database[scenario_index][cav_id][curr_timestamp_key]['lane']
        # transform = transforms.ToTensor()
        lane_image = Image.open(lane_path)
        lane_image = np.array(lane_image)
        lane_image = np.expand_dims(lane_image, axis=0)
        lane_image = np.repeat(lane_image, center_objects.shape[0], axis=0) # (16, 256, 256, 3)
        ret_dict['map_polylines'] = lane_image

        # Load Fused CoBEVT features.
        fused_feature_path = self.scenario_database[scenario_index][ego_cav_id][curr_timestamp_key]['fused_feature']
        fused_feature = np.load(fused_feature_path)  # [1, 256, 48, 176]
        ret_dict['fused_feature'] = fused_feature

        return ret_dict

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data. Each single piece of data corresponds to
         the -1/+5s trajectories at (scenario, cav_id, timestamp_key) plus the
         BEV feature at (scenario, cav_id, timestamp_key)

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

        """
        # Find the scenario index by looping over the cumulative length of records.
        scenario_index = self.records[idx]['scenario_idx']
        scenario_name = self.idx2scenrio[scenario_index].split('/')[-1]
        ego_cav_id = self.records[idx]['ego_cav_id']
        timestamp_key = self.records[idx]['timestamp_key']
        timestamp_idx = self.records[idx]['timestamp_idx']
        cav_ids_that_have_valid_data = self.records[idx]['cav_ids_that_have_valid_data']

        # Collect all CAV trajectories in the current scenario.
        cavs_gt_traj_data = []
        cavs_pred_traj_data = []
        cavs_timestamp_keys = []
        cavs_id = []
        for cav_id in cav_ids_that_have_valid_data:
            # Load the ground truth and pred trajs.
            with open(self.scenario_database[scenario_index][cav_id][timestamp_key]['preprocessed_gt_data'], 'rb') as f:
                ret = pickle.load(f)
                gt_data, timestamp_keys_gt = ret['data'], ret['timestamps']
            with open(self.scenario_database[scenario_index][cav_id][timestamp_key]['preprocessed_pred_data'],
                      'rb') as f:
                ret = pickle.load(f)
                pred_data, timestamp_keys_pred = ret['data'], ret['timestamps']

            # # Temp Hack since some gt dataset has vel. TODO: remove
            # for _, car_states in pred_data.items():
            #     for state in car_states:
            #         del state[-3:-1]
        
            common_timestamps_keys = list(
                set(timestamp_keys_gt).intersection(set(timestamp_keys_pred)))

            gt_traj_data = OrderedDict(sorted(gt_data.items(), key=lambda x: x[0]))
            pred_traj_data = OrderedDict(sorted(pred_data.items(), key=lambda x: x[0]))

            cavs_gt_traj_data.append(gt_traj_data)
            cavs_pred_traj_data.append(pred_traj_data)
            cavs_timestamp_keys.append(common_timestamps_keys)
            cavs_id.append(cav_id)

        # Loop through all CAVs again.
        cavs_data_for_mtr = []
        for gt_data, pred_data, cav_id, timestamp_keys in zip(cavs_gt_traj_data, cavs_pred_traj_data, cavs_id,
                                                              cavs_timestamp_keys):
            cavs_data_for_mtr.append(self.organize_trajectory_timestamp_into_dict(gt_data,
                                                                                  pred_data,
                                                                                  timestamp_keys,
                                                                                  scenario_index,
                                                                                  scenario_name,
                                                                                  cav_id,
                                                                                  ego_cav_id,
                                                                                  timestamp_key,
                                                                                  timestamp_idx))

        return cavs_data_for_mtr

    def collate_batch(self, batch_list):
        """
        Args:
        batch_list:
            scenario_id: (num_center_objects)
            scenario_name: (num_center_objects)
            timestamp_idx: (num_center_objects)
            timestamp_key: (num_center_objects)
            ego_cav_id: (num_center_objects)
            track_index_to_predict (num_center_objects):

            obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            obj_trajs_mask (num_center_objects, num_objects, num_timestamps):
            map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

            obj_trajs_pos: (num_center_objects, num_objects, num_timestamps, 3)
            obj_trajs_last_pos: (num_center_objects, num_objects, 3)
            obj_types: (num_objects)
            obj_ids: (num_objects)

            center_objects_world: (num_center_objects, 9)  [cx, cy, cz, dx, dy, dz, heading, vel, valid]
            center_objects_type: (num_center_objects)
            center_objects_id: (num_center_objects)

            obj_trajs_future_state (num_center_objects, num_objects, num_future_timestamps, 3): [x, y, vel]
            obj_trajs_future_mask (num_center_objects, num_objects, num_future_timestamps):
            center_gt_trajs (num_center_objects, num_future_timestamps, 3): [x, y, vel]
            center_gt_trajs_mask (num_center_objects, num_future_timestamps):
            center_gt_final_valid_idx (num_center_objects): the final valid timestamp in num_future_timestamps
        """
        cavs_data_for_mtr = batch_list[0]  # there is only one

        num_cavs = len(cavs_data_for_mtr)

        key_to_list = {}
        for key in cavs_data_for_mtr[0].keys():
            key_to_list[key] = [cav_data_for_mtr[key] for cav_data_for_mtr in cavs_data_for_mtr]

        input_dict = {}
        for key, val_list in key_to_list.items():

            if key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask', 'map_polylines_center',
                       'obj_trajs_pos', 'obj_trajs_last_pos', 'obj_trajs_future_state', 'obj_trajs_future_mask']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = common_utils.merge_batch_by_padding_2nd_dim(val_list)
            elif key in ['scenario_id', 'scenario_name', 'timestamp_idx', 'timestamp_key', 'ego_cav_id', 'obj_types', 'obj_ids',
                         'center_objects_type', 'center_objects_id']:
                input_dict[key] = np.concatenate(val_list, axis=0)
            elif key in ['fused_feature']:
                input_dict[key] = [torch.from_numpy(x) for x in val_list]
            else:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = torch.cat(val_list, dim=0)
        
        batch_sample_count = [len(x['track_index_to_predict']) for x in cavs_data_for_mtr]
        batch_dict = {'num_cavs': num_cavs,
                      'input_dict': input_dict,
                      'batch_sample_count': batch_sample_count}

        scenario_name = input_dict['scenario_name'][0]
        ego_cav_id = input_dict['ego_cav_id'][0]
        curr_timestamp_idx = input_dict['timestamp_idx'][0]

        return batch_dict

    @staticmethod
    def create_agent_data_for_center_objects(
            center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types, obj_ids
    ):
        obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = OPV2VMultiEgoDataset.generate_centered_trajs_for_agents(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_types=obj_types, center_indices=track_index_to_predict,
            sdc_index=sdc_track_index, timestamps=timestamps, obj_trajs_future=obj_trajs_future
        )

        # generate the labels of track_objects for training
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[
            center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps, 3)
        center_gt_trajs_mask = obj_trajs_future_mask[
            center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps)
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # filter invalid past trajs
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

        obj_trajs_mask = obj_trajs_mask[:,
                         valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
        obj_trajs_data = obj_trajs_data[:,
                         valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
        obj_trajs_future_state = obj_trajs_future_state[:,
                                 valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 2):  [x, y]
        obj_trajs_future_mask = obj_trajs_future_mask[:,
                                valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):
        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

        assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
        assert len(obj_types) == obj_trajs_future_mask.shape[1]
        assert len(obj_ids) == obj_trajs_future_mask.shape[1]

        # generate the final valid position of each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
            center_gt_final_valid_idx[cur_valid_mask] = k

        return (obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids)

    @staticmethod
    def get_interested_agents(track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            # assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'
            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:  # skip invalid state
                continue
            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    @staticmethod
    def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]
    
        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1,
                                                                                             1)

        # for x,y,z,l,w,h,heading,valid in obj_trajs[0,0,:,:]:
        #     print(format(x, 'f'), format(y, 'f'))
            
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]

        # for x,y,z,l,w,h,heading,valid in obj_trajs[0,0,:,:]:
        #     print(format(x, 'f'), format(y, 'f'))
        
        # print(center_heading)
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)


        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    @staticmethod
    def generate_centered_trajs_for_agents(center_objects, obj_trajs_past, obj_types, center_indices, sdc_index,
                                           timestamps, obj_trajs_future):
        """[summary]

        Args:
            center_objects (num_center_objects, 8): [x, y, z, l, w, h, heading, valid]
            obj_trajs_past (num_objects, num_timestamps, 8): [x, y, z, l, w, h, heading, valid]
            obj_types (num_objects):
            center_indices (num_center_objects): the index of center objects in obj_trajs_past
            centered_valid_time_indices (num_center_objects), the last valid time index of center objects
            timestamps ([type]): [description]
            obj_trajs_future (num_objects, num_future_timestamps, 8): [x, y, z, l, w, h, heading, valid]
        Returns:
            ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
            ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
            ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
        """
        assert obj_trajs_past.shape[-1] == 8
        assert center_objects.shape[-1] == 8
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        # transform to cpu torch tensor
        center_objects = torch.from_numpy(center_objects).float()
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        timestamps = torch.from_numpy(timestamps).float()

        # transform coordinates to the centered objects
        obj_trajs = OPV2VMultiEgoDataset.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6
        )

        ## generate the attributes for each object
        object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        # object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
        # object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1
        object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 1] = 1
        # object_onehot_mask[:, sdc_index, :, 2] = 1 # sdc_index?

        object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
        object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        # vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        # vel_pre = torch.roll(vel, shifts=1, dims=2)
        # acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        # acce[:, :, 0, :] = acce[:, :, 1, :]

        ret_obj_trajs = torch.cat((
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            # obj_trajs[:, :, :, 7:9],
            # acce,
        ), dim=-1)

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        ##  generate label for future trajectories
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_trajs_future = OPV2VMultiEgoDataset.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6
        )

        ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1]]  # (x, y)
        ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

        return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()

    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0,
                                          num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                       axis=-1)  # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = \
            (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4],
                            axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask

    @staticmethod
    def generate_prediction_dicts(batch_pred_dicts, output_path=None):
        """

        Args:
            batch_pred_dicts:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_pred_dicts['input_dict']

        pred_trajs = batch_pred_dicts['pred_trajs']
        pred_scores = batch_pred_dicts['pred_scores']

        pred_dict_list = []
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 5

        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

        this_ego_pred_dict_list = []
        batch_sample_count = batch_pred_dicts['batch_sample_count']
        start_obj_idx = 0
        for bs_idx in range(batch_pred_dicts['num_cavs']):
            cur_scene_pred_list = []
            for obj_idx in range(start_obj_idx, start_obj_idx + batch_sample_count[bs_idx]):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][obj_idx],
                    'scenario_name': input_dict['scenario_name'][obj_idx],
                    'timestamp_idx': input_dict['timestamp_idx'][obj_idx],
                    'timestamp_key': input_dict['timestamp_key'][obj_idx],
                    'ego_cav_id': input_dict['ego_cav_id'][obj_idx],
                    'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
                    # 'obj_trajs':  input_dict['obj_trajs'][obj_idx],
                    'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                    'object_id': input_dict['center_objects_id'][obj_idx],
                    'object_type': input_dict['center_objects_type'][obj_idx],
                    'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),
                    'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy(),
                    'center_gt_trajs_mask': input_dict['center_gt_trajs_mask'][obj_idx].cpu().numpy(),
                    'center_gt_final_valid_idx': input_dict['center_gt_final_valid_idx'][obj_idx].cpu().numpy().astype(
                        np.int64)
                }
                cur_scene_pred_list.append(single_pred_dict)

            this_ego_pred_dict_list.append(cur_scene_pred_list)
            start_obj_idx += batch_sample_count[bs_idx]

        assert start_obj_idx == num_center_objects
        assert len(this_ego_pred_dict_list) == batch_pred_dicts['num_cavs']
        pred_dict_list.extend(this_ego_pred_dict_list)

        return pred_dict_list

    def evaluation(self, pred_dicts, output_path=None, eval_method='waymo', **kwargs):
        if eval_method == 'waymo':
            from .waymo.waymo_eval import waymo_evaluation
            try:
                num_modes_for_eval = pred_dicts[0][0]['pred_trajs'].shape[0]
            except:
                num_modes_for_eval = 6
            metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts,
                                                                 num_modes_for_eval=num_modes_for_eval,
                                                                 eval_second=self.dataset_cfg.FUTURE_FRAMES // 10)

            metric_result_str = '\n'
            for key in metric_results:
                metric_results[key] = metric_results[key]
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            metric_result_str += '\n'
            metric_result_str += result_format_str
        else:
            raise NotImplementedError

        return metric_result_str, metric_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='MTR/output/v2v4real_multiego_gt/motion_aggregation/v2v4real_multiego_gt.yaml',
                        help='specify the config of dataset')
    args = parser.parse_args()

    import yaml
    from easydict import EasyDict

    try:
        yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)

    dataset = OPV2VMultiEgoDataset(dataset_cfg=dataset_cfg, class_names=['VEHICLE'], training=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
