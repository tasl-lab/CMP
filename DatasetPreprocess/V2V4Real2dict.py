import os
import yaml
import argparse
from pathlib import Path
from collections import OrderedDict
import json
import pickle
from PIL import Image
import numpy as np
import copy

from opencood.data_utils.v2v4real.datasets import GT_RANGE
from opencood.utils import box_utils_v2v4real as box_utils
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.hypes_yaml.yaml_utils import load_yaml

from tqdm import tqdm
import sys
sys.path.append("./MTR")
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets.v2v4real_multiego_dataset import V2V4RealMultiEgoDataset

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--traj_extraction', action='store_true', default=False)
    parser.add_argument('--output_path', type=str, default='./preprocessed_data',
                        help='output path for saving preprocessed data')
    parser.add_argument('--tag', type=str, default='gt',
                        help='tag for the preprocessed data')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def save_dictionary_to_file(dictionary, filename):
    # with open(filename, 'w') as file:
    #     json.dump(dictionary, file)
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)


def traj_extraction_multiego(scenario_database, output_dir, idx2scenrio):
    for scenario_idx, scenario_cavs in tqdm(scenario_database.items()):
        
        ego_lidar_pose = []
        cav_lidar_pose = []

        for cav_id, content in scenario_cavs.items():  # only extract surrounding cars of the ego car?
            timestamps = [key for key in content.keys() if key.startswith('0')]
            
            for i, timestamp in enumerate(timestamps):
                cur_params = load_yaml(content[timestamp]['yaml'])
                if cav_id == '0':
                    ego_lidar_pose.append(cur_params['lidar_pose'])
                else:
                    cav_lidar_pose.append(cur_params['lidar_pose'])

        assert len(ego_lidar_pose) == len(timestamps) 
        assert len(cav_lidar_pose) == len(timestamps) 
        
        for cav_id, content in scenario_cavs.items():  # only extract surrounding cars of the ego car?

            if cav_id == '1':
                continue
                 
            data = OrderedDict()
            timestamps = [key for key in content.keys() if key.startswith('0')]
            
            for i, timestamp in enumerate(timestamps):
                cur_params = load_yaml(content[timestamp]['yaml'])
                car_manipulated = dict.fromkeys(data.keys(), 0)

                if cav_id == '0':
                    # iterate all neighbors.
                    for vehicle_id, vehicle_params in cur_params['vehicles'].items():
                        if vehicle_params['obj_type'] != 'Car':
                            continue

                        if vehicle_id not in data.keys():
                            data[vehicle_id] = []
                            for j in range(i):
                                data[vehicle_id].append(
                                    [0, 0, 0, 0, 0, 0, 0, 0])  # padding invalid state for the beginning
                                    
                        roll, yaw, pitch = vehicle_params['angle']
                        xyz = [loc + cen for loc, cen in zip(vehicle_params['location'], vehicle_params['center'])]
                        l, w, h = vehicle_params['extent']
                        data[vehicle_id].append([*xyz, 2*l, 2*w, 2*h, vehicle_params['angle'][1],
                                                1])  # x, y, z, l, w, h, theta, valid
                        car_manipulated[vehicle_id] = 1
                else:
                    for vehicle_id, vehicle_params in cur_params['vehicles'].items():
                        if vehicle_params['obj_type'] != 'Car':
                            continue
                        invalid_bbx_lidar = False
                        location = vehicle_params['location']
                        rotation = vehicle_params['angle']
                        center = vehicle_params['center']
                        extent = vehicle_params['extent']
                        object_pose = [location[0] + center[0], location[1] + center[1], location[2] + center[2], rotation[0], rotation[1], rotation[2]]
                        # print(object_pose)
                        transformation_matrix = x1_to_x2(cav_lidar_pose[i], ego_lidar_pose[i])
                        object2lidar = x1_to_x2(object_pose, transformation_matrix)
                        # shape (3, 8)
                        bbx = box_utils.create_bbx(extent).T
                        # bounding box under ego coordinate shape (4, 8)
                        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

                        # project the 8 corners to world coordinate
                        bbx_lidar = np.dot(object2lidar, bbx).T
                        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
                        bbx_lidar = box_utils.corner_to_center(bbx_lidar, order='hwl')

                        if len(bbx_lidar) > 0 and len(bbx_lidar[0]) == 7:
                            bbx_lidar = bbx_lidar[0]
                        else:
                            print("Invalid bbx_lidar")
                            invalid_bbx_lidar = True
                        if vehicle_params['ass_id'] != -1:
                            vehicle_id = vehicle_params['ass_id']
                        else:
                            vehicle_id += 1000 * int(cav_id)
                        
                        if vehicle_id not in data.keys():
                            data[vehicle_id] = []
                            for j in range(i):
                                data[vehicle_id].append(
                                    [0, 0, 0, 0, 0, 0, 0, 0]) 

                        if invalid_bbx_lidar:
                            data[vehicle_id].append([0,0,0,0,0,0,0,0])  # x, y, z, l, w, h, theta, valid
                        else:
                            data[vehicle_id].append([bbx_lidar[0], bbx_lidar[1], bbx_lidar[2], bbx_lidar[5], bbx_lidar[4], bbx_lidar[3], np.rad2deg(bbx_lidar[6]), 1])  # x, y, z, l, w, h, theta, valid

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

            data = V2V4RealMultiEgoDataset.interpolate_car_info(data)    
            if data == None:
                print("Data is None")
                continue
        
            save_dict = {'data': data, 'timestamps': timestamps}
            save_dictionary_to_file(save_dict, os.path.join(output_dir, f"{idx2scenrio[scenario_idx].split('/')[-1]}-{cav_id}-traj.pickle"))

def GetGTForSplit(is_train):
    print("Processing GT for split " + ("train" if is_train else "test"))

    args, cfg = parse_config()
    output_dir = os.path.join(args.output_path, args.tag, 'train' if is_train else 'test')
    os.makedirs(output_dir, exist_ok=True)
    train_dir = cfg.DATA_CONFIG.train_dir if is_train else cfg.DATA_CONFIG.validate_dir
    params = cfg.DATA_CONFIG
    # first load all paths of different scenarios
    scenario_folders = sorted([os.path.join(train_dir, x)
                               for x in os.listdir(train_dir) if
                               os.path.isdir(os.path.join(train_dir,
                                                          x)) and x != '2021_09_09_13_20_58'])  # 2021_09_09_13_20_58 doesn't have bev lane images

    scenario_database = OrderedDict()
    idx2scenrio = {}
    len_record = []

    if 'max_cav' not in params:
        max_cav = 7
    else:
        max_cav = params['max_cav']

    # loop over all scenarios
    for (i, scenario_folder) in enumerate(scenario_folders):
        scenario_database.update({i: OrderedDict()})
        idx2scenrio[i] = scenario_folder
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
            # if j > max_cav - 1:
            #     print('too many cavs')
            #     break
            scenario_database[i][cav_id] = OrderedDict()

            # save all yaml files to the dictionary
            cav_path = os.path.join(scenario_folder, cav_id)

            # use the frame number as key, the full path as the values
            yaml_files = \
                sorted([os.path.join(cav_path, x)
                        for x in os.listdir(cav_path) if
                        x.endswith('.yaml') and 'additional' not in x])
            timestamps = V2V4RealMultiEgoDataset.extract_timestamps(yaml_files)

            for timestamp in timestamps:
                scenario_database[i][cav_id][timestamp] = \
                    OrderedDict()

                yaml_file = os.path.join(cav_path,
                                         timestamp + '.yaml')
                lidar_file = os.path.join(cav_path,
                                          timestamp + '.pcd')

                scenario_database[i][cav_id][timestamp]['yaml'] = \
                    yaml_file
                scenario_database[i][cav_id][timestamp]['lidar'] = \
                    lidar_file

            # Assume all cavs will have the same timestamps length. Thus
            # we only need to calculate for the first vehicle in the
            # scene.
            if j == 0:
                # we regard the agent with the minimum id as the ego
                scenario_database[i][cav_id]['ego'] = True
                if not len_record:
                    len_record.append(len(timestamps))
                else:
                    prev_last = len_record[-1]
                    len_record.append(prev_last + len(timestamps))
            else:
                scenario_database[i][cav_id]['ego'] = False

    if args.traj_extraction:
        traj_extraction(scenario_database, output_dir, idx2scenrio)

    traj_extraction_multiego(scenario_database, output_dir, idx2scenrio)

if __name__ == "__main__":
    GetGTForSplit(is_train=True)
    GetGTForSplit(is_train=False)