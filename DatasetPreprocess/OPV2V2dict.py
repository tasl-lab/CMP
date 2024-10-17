import os
import yaml
import argparse
from pathlib import Path
from collections import OrderedDict
import json
import pickle
from PIL import Image
import numpy as np

import sys

from tqdm import tqdm

sys.path.append("./MTR")
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets.opv2v_multiego_dataset import OPV2VMultiEgoDataset


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--traj_extraction', action='store_true', default=False)
    parser.add_argument('--bev_lane_extraction', action='store_true', default=False)
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
        for cav_id, content in scenario_cavs.items():  # only extract surrounding cars of the ego car?
            data, timestamps = OPV2VMultiEgoDataset.get_gt_traj(content)
            save_dict = {'data': data, 'timestamps': timestamps}
            save_dictionary_to_file(save_dict,
                                    os.path.join(output_dir, f"{idx2scenrio[scenario_idx].split('/')[-1]}-{cav_id}-traj.pickle"))



def bev_lane_extraction(scenario_database, output_dir, idx2scenrio):
    for i, content in tqdm(scenario_database.items()):
        scenario = scenario_database[i]
        image_list = OrderedDict()
        for cav_id, cav_content in scenario.items():
            image_list[cav_id] = OrderedDict()
            timestamps = list(cav_content.keys())
            for timestamp in timestamps:
                if timestamp == 'ego': continue

                lane_path = scenario[cav_id][timestamp]['lane']
                if os.path.exists(lane_path):
                    lane_image = Image.open(lane_path)
                else:
                    continue

                lane_image = np.array(lane_image)
                # lane_image = lane_image.tolist()
                image_list[cav_id][timestamp] = lane_image

        save_dictionary_to_file(image_list,
                                os.path.join(output_dir, f"{idx2scenrio[i].split('/')[-1]}-bev_lane_images.pickle"))


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

        additional_root = params['additional_dir']
        lane_path = os.path.join(additional_root, 'train' if is_train else 'test')

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
            timestamps = OPV2VMultiEgoDataset.extract_timestamps(yaml_files)

            for timestamp in timestamps:
                scenario_database[i][cav_id][timestamp] = \
                    OrderedDict()

                yaml_file = os.path.join(cav_path,
                                         timestamp + '.yaml')
                lidar_file = os.path.join(cav_path,
                                          timestamp + '.pcd')
                camera_files = OPV2VMultiEgoDataset.load_camera_files(cav_path, timestamp)
                lane_file = os.path.join(lane_path, scenario_folder.split('/')[-1], cav_id, f"{timestamp}_bev_lane.png")

                scenario_database[i][cav_id][timestamp]['yaml'] = \
                    yaml_file
                scenario_database[i][cav_id][timestamp]['lidar'] = \
                    lidar_file
                scenario_database[i][cav_id][timestamp]['camera0'] = \
                    camera_files
                scenario_database[i][cav_id][timestamp]['lane'] = \
                    lane_file
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

    if args.bev_lane_extraction:
        bev_lane_extraction(scenario_database, output_dir, idx2scenrio)

if __name__ == "__main__":
    GetGTForSplit(is_train=True)
    GetGTForSplit(is_train=False)