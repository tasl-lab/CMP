import argparse
import copy
import os
import time
import traceback
from collections import OrderedDict, Counter

from tqdm import tqdm
import numpy as np
import pickle
import torch
import open3d as o3d
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils import box_utils_v2v4real as box_utils
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.utils import eval_utils
from opencood.utils.box_utils_v2v4real import corner_to_standup_box_torch, corner_to_center
from opencood.utils.box_utils import inverse_project_world_objects

import opencood.hypes_yaml.yaml_utils as yaml_utils

from opencood.tools import train_utils, inference_utils

from opencood.data_utils.opv2v.datasets import build_opv2v_dataset
from opencood.data_utils.v2v4real.datasets import build_v2v4real_dataset
from opencood.data_utils.v2v4real.datasets import GT_RANGE

from opencood.visualization import vis_utils

from multi_ego_inference_utils import interpolate_speed, interpolate_states

from AB3DMOT_libs.model import AB3DMOT
from AB3DMOT_libs.utils import pose_to_transformation_matrix
from AB3DMOT_libs.io import load_detection, get_saving_dir, get_frame_det, save_results, save_affinity

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (50000, rlimit[1]))

def parse_arguments():
    parser = argparse.ArgumentParser(description='multi ego inference configuration')
    
    parser.add_argument('--use_train_set', action='store_true', 
                        help='Flag to use the training set (default: False)')
    parser.add_argument('--remove_tracking_speed', action='store_false', 
                        help='Flag to remove tracking speed (default: True)')
    parser.add_argument('--perception_model_name', type=str, required=True, 
                        help='Name of the perception model to use (required)')
    parser.add_argument('--no_coop', action='store_true', 
                        help='Flag to use no cooperative setting(default: False)')

    args = parser.parse_args()
    return args

def ConvectSingleFrameDetectionsToAB3DMOTFormat(all_detections_for_this_frame):
    """
    @param detections_list_for_this_frame list of detections:
                {
                    'timestamp_idx': timestamp_idx,
                    'standup_box_np': standup_box_np,  # xmin, ymin, xmax, ymax
                    'pred_score_np': pred_score_np,  # x, y, z, l, w, h, ry
                    'pred_box3d_np': pred_box3d_np,
                    'result_stat': result_stat,
                    'speeds': speeds,
                }
    @return list of strings in AB3DMOT format, each string is a single detection.
    """
    strings = []
    num_detections = len(all_detections_for_this_frame['pred_box3d_np'])
    timestamp_idx = all_detections_for_this_frame['timestamp_idx']
    standup_box_np = all_detections_for_this_frame['standup_box_np']
    pred_score_np = all_detections_for_this_frame['pred_score_np']
    pred_box3d_np = all_detections_for_this_frame['pred_box3d_np']
    matched_car_id = all_detections_for_this_frame['matched_car_id']
    speeds = all_detections_for_this_frame['speeds']

    for l in range(num_detections):
        # timestamp, obj_type(2 for car), xmin, ymin, xmax, ymax, score, x, y, z, l, w, h, theta, orit, matched_car_id, vel_x, vel_y
        ab3dmot_input_string = f"{timestamp_idx},2," \
                               f"{standup_box_np[l][0]}," \
                               f"{standup_box_np[l][2]}," \
                               f"{standup_box_np[l][1]}," \
                               f"{standup_box_np[l][3]}," \
                               f"{pred_score_np[l]}," \
                               f"{pred_box3d_np[l][5]}," \
                               f"{pred_box3d_np[l][4]}," \
                               f"{pred_box3d_np[l][3]}," \
                               f"{pred_box3d_np[l][0]}," \
                               f"{pred_box3d_np[l][1]}," \
                               f"{pred_box3d_np[l][2]}," \
                               f"{pred_box3d_np[l][6]}," \
                               f"{pred_box3d_np[l][6]}," \
                               f"{matched_car_id[l]}," \
                               f"{speeds[l][1]}," \
                               f"{speeds[l][0]}\n"
        strings.append(ab3dmot_input_string)

    return strings


def PerformDetectionWithPretrained(path_to_tracking_label_save=None,
                                    path_to_perception_pretrained=None,
                                    path_to_detection_output=None,
                                    path_to_detection_fused_features=None,
                                    path_to_mtr_traj_gt=None,
                                    dataset_name='opv2v',
                                    use_train_set=False,
                                    save_perception_features=False,
                                    no_coop=False,
                                    show_visualization=False):
    """
    Perform detection with pretrained perception model.
    Note that you need to specify the path to data in the config.yaml file in the pretrained model folder.
    """

    # When using pretrained model, we load hyperparameters from pretrained model.
    hypes = yaml_utils.load_yaml(path_to_perception_pretrained + '/config.yaml')

    # Build Dataset.
    if dataset_name == 'opv2v':
        opencood_dataset = build_opv2v_dataset(hypes, visualize=True, train=use_train_set)
    elif dataset_name == 'v2v4real':
        opencood_dataset = build_v2v4real_dataset(hypes, visualize=True, train=use_train_set, isSim=False)
    else:
        assert dataset_name in ['opv2v', 'v2v4real'], f"{dataset_name} is not found. Should be either 'opv2v' or 'v2v4real'"
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=12,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    # Create and load the model.
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = path_to_perception_pretrained
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Allocate perception results. Per scenario, per who_is_ego, per timestamp.
    detection_result_for_tracking = {}

    result_stat_all = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'matched_indices': []},
                       0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'matched_indices': []},
                       0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'matched_indices': []}}

    previous_id = None

    # Inference.
    print('Running Inference')
    for batch_data in tqdm(data_loader):
        scenario_path = batch_data['ego']['scenario_list'][0]  # A path to scenario folder.
        scenario_name = scenario_path.split('/')[-1]
        timestamp_key = batch_data['ego']['timestamp_key_list'][0]
        timestamp_idx = batch_data['ego']['timestamp_idx_list'][0]
        ego_cav_id = batch_data['ego']['ego_cav_id_list'][0]
        cur_lidar_pose = batch_data['ego']['lidar_pose'][0]  # (6,) or (4,4)
        transformation_matrix = None
        if not no_coop and dataset_name == 'v2v4real':
            transformation_matrix = batch_data['ego']['transformation_matrix_list'][0]

        if previous_id != (scenario_name, ego_cav_id):
            car_manipulated = OrderedDict()
            previous_id = (scenario_name, ego_cav_id)

        if scenario_name == '2021_09_09_13_20_58':
            continue

        # print(f"Processing {scenario_name} when ego is {ego_cav_id} at time {timestamp_key}(Index: {timestamp_idx}). ")
        curr_path = os.path.join(path_to_detection_output, scenario_name)  # Put all vehicle's results in one folder.
        os.makedirs(curr_path, exist_ok=True)

        result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'matched_indices': []},
                       0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'matched_indices': []},
                       0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': [], 'matched_indices': []}}

        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            
            if not no_coop:
                pred_box_tensor, pred_box3d, pred_score, gt_box_tensor, gt_object_id_tensor, output_dict = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                for_tracking=True)
            else:
                pred_box_tensor, pred_box3d, pred_score, gt_box_tensor, gt_object_id_tensor, output_dict = \
                    inference_utils.inference_no_fusion(batch_data,
                                                        model,
                                                        opencood_dataset,
                                                        for_tracking=True)
        

            if pred_box_tensor is None:
                print("WARN: No detection found.")
                continue

            # Convert to 2D axis aligned box. (xmin, ymin, xmax, ymax)
            standup_box = corner_to_standup_box_torch(pred_box_tensor)
            pred_box_np, pred_box3d_np, pred_score_np, standup_box_np, cur_lidar_pose_np = map(
                lambda x: x.cpu().detach().numpy(),
                (pred_box_tensor, pred_box3d, pred_score, standup_box, cur_lidar_pose)
            )

            # Find confidence scores.
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3,
                                       gt_object_id_tensor)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5,
                                       gt_object_id_tensor)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7,
                                       gt_object_id_tensor)
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_all,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_all,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_all,
                                       0.7)
                                     
            # Interpolate speed.
            speeds = interpolate_speed(pred_box3d_np)

            # Generate results.
            all_detections_for_this_frame = {
                'timestamp_idx': timestamp_idx,
                'timestamp_key': timestamp_key,
                'cur_lidar_pose_np': cur_lidar_pose_np,
                'standup_box_np': standup_box_np,  # xmin, ymin, xmax, ymax
                'pred_score_np': pred_score_np,  # x, y, z, l, w, h, ry
                'pred_box3d_np': pred_box3d_np,
                'matched_car_id': result_stat[0.7]['matched_indices'][:len(pred_score_np)],
                'speeds': speeds,
                'transformation_matrix': transformation_matrix
            }
            # Save immediate perception features to disk.
            if save_perception_features:
                if os.path.exists(os.path.join(path_to_detection_fused_features, f"fused_feature_{scenario_name}_{ego_cav_id}_{timestamp_idx}.pkl")):
                    continue
                os.makedirs(path_to_detection_fused_features, exist_ok=True)
                if no_coop:
                    fused_feature_np = output_dict['ego']['spatial_features_2d'].cpu().detach().numpy()
                else:
                    fused_feature_np = output_dict['ego']['fused_feature'].cpu().detach().numpy()
                np.save(os.path.join(path_to_detection_fused_features, f"fused_feature_{scenario_name}_{ego_cav_id}_{timestamp_idx}.pkl"),
                        fused_feature_np)
                all_detections_for_this_frame['path_to_fused_feature'] = os.path.join(path_to_detection_fused_features, f"fused_feature_{scenario_name}_{ego_cav_id}_{timestamp_idx}.pkl")

            # Save to memory.
            if scenario_name not in detection_result_for_tracking:
                detection_result_for_tracking[scenario_name] = {}
            if ego_cav_id not in detection_result_for_tracking[scenario_name]:
                detection_result_for_tracking[scenario_name][ego_cav_id] = {}
            if timestamp_idx not in detection_result_for_tracking[scenario_name][ego_cav_id]:
                detection_result_for_tracking[scenario_name][ego_cav_id][timestamp_idx] = {}
            detection_result_for_tracking[scenario_name][ego_cav_id][timestamp_idx] = all_detections_for_this_frame

            # Save detections to disk.
            if path_to_detection_output is not None:
                with open(os.path.join(curr_path, f"{ego_cav_id}.txt"), 'a') as file:
                    for line in ConvectSingleFrameDetectionsToAB3DMOTFormat(all_detections_for_this_frame):
                        file.write(line)

            # Visualize
            if show_visualization:
                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  True,
                                                  "",
                                                  dataset=opencood_dataset)

            # Quick tracker eval.
            eval_utils.eval_final_results(result_stat, saved_path)

    eval_utils.eval_final_results(result_stat_all, saved_path)

    return detection_result_for_tracking


def TrackByAB3DMOT(detection_result_for_tracking,
                   path_to_tracking_output=None,
                   path_to_ab3dmot_file_output=None,
                   remove_tracking_speed=True,
                   dataset_name='opv2v',
                   no_coop=False):
    # Build logger.
    os.makedirs(path_to_tracking_output, exist_ok=True)
    log_file = open(os.path.join(path_to_tracking_output, 'log.txt'), 'w')

    # Build config.
    tracker_config = edict({
        'dataset': dataset_name,
        'score_threshold': -10000,
        # filter out tracklet with low confidence if necessary, default no filtering here but do it in trk_conf_threshold.py
        'num_hypo': 1,  # >1 to allow multi-hypothesis tracking
        'ego_com': True,  # turn on only slightly reduce speed but increase a lot for performance
        'vis': False,  # only for debug or visualization purpose, will significantly reduce speed
        'affi_pro': True
    })

    # Allocate tracking results. Per scenario, per who_is_ego.
    tracking_result_for_prediction = {}

    # AB3DMOT stat
    eval_dir_dict = dict()
    eval_dir_dict[0] = os.path.join(path_to_ab3dmot_file_output, 'data_%d' % 0)
    os.makedirs(eval_dir_dict[0], exist_ok=True)

    # Iterate the instances.
    for scenario_name, scenario_to_ego_cav_dict in tqdm(detection_result_for_tracking.items()):
        for ego_cav_id, ego_cav_to_timestamp_dict in scenario_to_ego_cav_dict.items():

            # Collect the ego poses.
            imu_poses = []
            timestamp_keys = []
            for timestamp_idx in sorted(ego_cav_to_timestamp_dict):
                all_detections_for_this_frame = ego_cav_to_timestamp_dict[timestamp_idx]
                cur_lidar_pose_np = all_detections_for_this_frame['cur_lidar_pose_np']
                imu_poses.append(pose_to_transformation_matrix(
                    np.expand_dims(cur_lidar_pose_np, axis=0)))  # Callee expects a time dimension at shape[0]
                timestamp_keys.append(all_detections_for_this_frame['timestamp_key'])
            
            imu_poses = np.concatenate(imu_poses, axis=0)  # Merge time dimension.

            # Initialize tracker for this time series.
            tracker = AB3DMOT(tracker_config, 'Car', calib=None, oxts=imu_poses,
                              img_dir=None, vis_dir=None,
                              hw={'image': (375, 1242), 'lidar': (720, 1920)},  # hw is not used
                              log=log_file, ID_init=1)

            # Iterate the time series.
            results_list = []
            ego_lidar_pose_list = []
            transformation_matrix_list = []
            for timestamp_idx in sorted(ego_cav_to_timestamp_dict):
                all_detections_for_this_frame = ego_cav_to_timestamp_dict[timestamp_idx]

                # Load detections.
                num_detections = len(all_detections_for_this_frame['pred_box3d_np'])
                standup_box_np = all_detections_for_this_frame['standup_box_np']
                pred_score_np = all_detections_for_this_frame['pred_score_np']
                pred_box3d_np = all_detections_for_this_frame['pred_box3d_np']  # n, (l, w, h, z, y, x, yaw)
                matched_car_id = all_detections_for_this_frame['matched_car_id']
                speeds = all_detections_for_this_frame['speeds']  # n, (vy, vx)
                cur_lidar_pose_np = all_detections_for_this_frame['cur_lidar_pose_np']
                transformation_matrix = None
                if not no_coop and dataset_name == 'v2v4real':
                    transformation_matrix = all_detections_for_this_frame['transformation_matrix']
                
                matched_car_id = np.array(matched_car_id)  
                mask = matched_car_id != -1
                pred_box3d_np = pred_box3d_np[mask]
                matched_car_id = matched_car_id[mask]
                standup_box_np = standup_box_np[mask]
                pred_score_np = pred_score_np[mask]
                num_detections = len(matched_car_id)

                # Convert to AB3DMOT format.
                dets_frame = {'dets': pred_box3d_np[:, [5, 4, 3, 0, 1, 2, 6]],  # n, (h, w, l, x, y, z, theta)
                              'info': np.stack([pred_box3d_np[:, 6],  # or0]entation
                                                    np.repeat(np.array([2]), num_detections),  # obj_type Car=2
                                                    standup_box_np[:, 0],  # xmin
                                                    standup_box_np[:, 2],  # ymin
                                                    standup_box_np[:, 1],  # xmax
                                                    standup_box_np[:, 3],  # ymax
                                                    pred_score_np], axis=1  # score
                                                ), 
                              'cav_id': matched_car_id,
                              'vel': speeds[:, [1, 0]]}  # n, (vx, vy)

                # Advance the tracker.
                # results are in (h,w,l,x,y,z,theta, obj_id, other info, confidence, cav_id)
                try:
                    frame_result, affi = tracker.track(dets_frame, timestamp_idx,
                                                  f'{scenario_name}_{ego_cav_id}_{all_detections_for_this_frame["timestamp_key"]}')
                except Exception:
                    print(f"Error when tracking {scenario_name}_{ego_cav_id}_{all_detections_for_this_frame['timestamp_key']}")
                    print(traceback.format_exc())
                    continue

                results_list.append(frame_result[0])  # Extract hypo 0.
                ego_lidar_pose_list.append(cur_lidar_pose_np)
                transformation_matrix_list.append(transformation_matrix)

                # File
                # saving affinity matrix, between the past frame and current frame
                # e.g., for 000006.npy, it means affinity between frame 5 and 6
                # note that the saved value in affinity can be different in reality because it is between the 
                # original detections and ego-motion compensated predicted tracklets, rather than between the 
                # actual two sets of output tracklets
                eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = \
                            get_saving_dir(eval_dir_dict, scenario_name, path_to_ab3dmot_file_output, 1, ego_cav_id)

                save_affi_file = os.path.join(affinity_dir, '%06d.npy' % timestamp_idx)
                save_affi_vis  = os.path.join(affinity_vis, '%06d.txt' % timestamp_idx)
                if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0): 
                    # save affinity as long as there are tracklets in at least one frame
                    np.save(save_affi_file, affi)

                    # cannot save for visualization unless both two frames have tracklets
                    if affi.shape[0] > 0 and affi.shape[1] > 0:
                        save_affinity(affi, save_affi_vis)

                # saving trajectories, loop over each hypothesis
                save_trk_file = os.path.join(save_trk_dir[0], '%06d.txt' % timestamp_idx)
                save_trk_file = open(save_trk_file, 'w')
                for line in frame_result[0]:
                    save_results(line, save_trk_file, eval_file_dict[0], \
                        {2: 'Car'}, timestamp_idx, -10000)
                save_trk_file.close()

            # Combine tracking results for the entire time horizon. See AB3Dmot2dict.py
            data_dict = {}
            cav_ids = {}
            for timestamp_idx, (frame_result, ego_lidar_pose, transformation_matrix) in enumerate(zip(results_list, ego_lidar_pose_list, transformation_matrix_list)):
                car_manipulated = dict.fromkeys(data_dict.keys(), 0)
                
                for detection_item in frame_result:
                    obj_id = int(detection_item[7])
                    cav_id = int(detection_item[-1])
                    confidence = detection_item[-2]
                    x = detection_item[3]
                    y = detection_item[4]
                    z = detection_item[5]
                    l = detection_item[2]
                    w = detection_item[1]
                    h = detection_item[0]
                    theta = detection_item[6]
                    
                    bbox_ego_coord = OrderedDict()
                    bbox_ego_coord[obj_id] = np.expand_dims(np.array([x, y, z, l, w, h, theta]), 0)
                    bbox_world_coord = inverse_project_world_objects(bbox_ego_coord, ego_lidar_pose, 'hwl')
                    car_state = [*list(bbox_world_coord.values())[0], 0, 0, confidence, 1]  ##placeholder for v_x and v_y, update later, the last 1 means valid
                    
                    # ID association.
                    if obj_id not in data_dict:
                        data_dict[obj_id] = []
                        for j in range(timestamp_idx):
                            data_dict[obj_id].append(
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # padding invalid state for the beginning
                        data_dict[obj_id].append(car_state)
                        cav_ids[obj_id] = []
                        cav_ids[obj_id].append(cav_id)
                    else:
                        data_dict[obj_id].append(car_state)
                        cav_ids[obj_id].append(cav_id)

                    car_manipulated[obj_id] = 1

                for vehicle_id, is_operated in car_manipulated.items():
                    if is_operated == 0:
                        data_dict[vehicle_id].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # padding invalid state for the end

            # padding past
            for car_id, timestamp_content in data_dict.items():
                idx = None
                for i in range(0, len(results_list) - 2):
                    if timestamp_content[i][-1] == 0 and timestamp_content[i + 1][-1] == 1:
                        idx = i
                        break
                    elif timestamp_content[i][-1] == 0 and timestamp_content[i + 1][-1] == 0:
                        continue
                    elif timestamp_content[i][-1] == 1:
                        break

                if idx is not None:
                    for j in range(idx, -1, -1):
                        data_dict[car_id][j] = copy.deepcopy(data_dict[car_id][j + 1])
                        data_dict[car_id][j][-4] = 0  # v_x
                        data_dict[car_id][j][-3] = 0  # v_y
                        data_dict[car_id][j][-2] = 0  # confidence
                        data_dict[car_id][j][-1] = 0  # valid

            # convert obj_id obtained from AB3Dmot to cav_id annotated in OPV2V
            converted_data = OrderedDict()  # car_id->car_states

            for obj_id, timestamp_content in data_dict.items():
                cur_cav_id = cav_ids[obj_id]
                counter = Counter(cur_cav_id)
                most_common = counter.most_common(1)
                cav_id = most_common[0][0]
                if cav_id == -1:
                    neg_obj_id = -obj_id
                    converted_data[neg_obj_id] = timestamp_content
                    continue
                if cav_id in converted_data.keys():
                    cur_frames = most_common[0][1]  # count how many frames this car_id appears
                    existing_frames = 0          # count how many frames this car_id appears in converted_data
                    for state in converted_data[cav_id]:
                        if state[-1] == 1:
                            existing_frames += 1
                    if cur_frames > existing_frames:
                        converted_data[cav_id] = timestamp_content
                else:
                    converted_data[cav_id] = timestamp_content

            # modify the keys of converted_data in the ascending order
            sorted_data = OrderedDict(sorted(converted_data.items()))

            data = sorted_data
            # data = data_dict
            # interpolate_velocity(data)

            # Optionally, remove speed.
            if remove_tracking_speed:
                for car_id, car_states in data.items():
                    for state in car_states:
                        del state[-4:-1]
                        assert len(state) == 8
        
            # Save tracking results.
            save_dict = {'data': data, 'timestamps': timestamp_keys}
            if scenario_name not in tracking_result_for_prediction:
                tracking_result_for_prediction[scenario_name] = {}
            if ego_cav_id not in tracking_result_for_prediction[scenario_name]:
                tracking_result_for_prediction[scenario_name][ego_cav_id] = {}
            tracking_result_for_prediction[scenario_name][ego_cav_id] = save_dict

            # Save to disk.
            if path_to_tracking_output is not None:
                with open(os.path.join(path_to_tracking_output, f"{scenario_name}-{ego_cav_id}-traj.pickle"), 'wb') as file:
                    pickle.dump(save_dict, file)
                    
    return tracking_result_for_prediction

if __name__ == '__main__':

    args = parse_arguments()
    project_dir = os.getcwd()
    dataset = 'opv2v'

    path_to_perception_pretrained = os.path.join(project_dir, 'pretrained', dataset, args.perception_model_name)
    path_to_detection_fused_features = os.path.join(project_dir, 'preprocessed_data', dataset, 'fused_features_' + args.perception_model_name)
    path_to_detection_output = os.path.join(project_dir, 'preprocessed_data', dataset, 'detection_output_' + args.perception_model_name, ('train' if args.use_train_set else 'test'))
    path_to_tracking_output = os.path.join(project_dir, 'preprocessed_data', dataset,'tracking_trajs_' + args.perception_model_name, ('train' if args.use_train_set else 'test'))
    path_to_ab3dmot_file_output = os.path.join(project_dir, 'preprocessed_data', dataset, 'ab3dmot_logs_' + args.perception_model_name, ('train' if args.use_train_set else 'test'))
    path_to_mtr_traj_gt = os.path.join(project_dir, 'preprocessed_data', dataset, 'gt' , ('train' if args.use_train_set else 'test'))
    path_to_tracking_label_save = os.path.join(project_dir, f'AB3Dmot/scripts/KITTI/{dataset}_label/', ('train' if args.use_train_set else 'test'))

    ## Stage 1: Detection.
    path_to_detection_cache = os.path.join(project_dir, 'preprocessed_data' , dataset) + '/' + args.perception_model_name + f'_detection_result_for_tracking_{"train" if args.use_train_set else "test"}.pkl'
    if not os.path.exists(path_to_detection_cache):
        print("Cached detection result not found. Performing detection with pretrained perception model.")
        detection_result_for_tracking = PerformDetectionWithPretrained(path_to_tracking_label_save=path_to_tracking_label_save,path_to_perception_pretrained=path_to_perception_pretrained, path_to_detection_output=path_to_detection_output, path_to_detection_fused_features=path_to_detection_fused_features, path_to_mtr_traj_gt=path_to_mtr_traj_gt, dataset_name=dataset, use_train_set=args.use_train_set, save_perception_features=True, no_coop=args.no_coop)
        os.makedirs(os.path.dirname(path_to_detection_cache), exist_ok=True)
        with open(path_to_detection_cache, 'wb') as file:
            pickle.dump(detection_result_for_tracking, file)
        print("Saved detection result to disk at " + path_to_detection_cache)
    else:
        print("Cached detection result found. Loading from disk. To recompute, simply delete " + path_to_detection_cache)
        with open(path_to_detection_cache, 'rb') as file:
            detection_result_for_tracking = pickle.load(file)

    ## Stage 2: Tracking.
    path_to_tracking_cache = os.path.join(project_dir, 'preprocessed_data' , dataset) + '/' + args.perception_model_name + f'_tracking_result_for_prediction_{"train" if args.use_train_set else "test"}.pkl'
    if not os.path.exists(path_to_tracking_cache):
        print("Cached tracking result not found. Performing tracking with AB3DMOT.")
        tracking_result_for_prediction = TrackByAB3DMOT(detection_result_for_tracking, path_to_tracking_output=path_to_tracking_output, path_to_ab3dmot_file_output=path_to_ab3dmot_file_output, remove_tracking_speed=args.remove_tracking_speed, dataset_name=dataset, no_coop=args.no_coop)
        os.makedirs(os.path.dirname(path_to_tracking_cache), exist_ok=True)
        with open(path_to_tracking_cache, 'wb') as file:
            pickle.dump(tracking_result_for_prediction, file)
        print("Saved tracking result to disk at " + path_to_tracking_cache)
    else:
        print(
            "Cached tracking result found. Loading from disk. To recompute, simply delete " + path_to_tracking_cache)
        with open(path_to_detection_cache, 'rb') as file:
            tracking_result_for_prediction = pickle.load(file)