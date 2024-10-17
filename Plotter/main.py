import glob
import os.path
from collections import OrderedDict
from pathlib import Path
from functools import reduce
import pylab
import argparse
from pathlib import Path

import numpy as np
import pickle
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from tqdm import tqdm

from opencood.hypes_yaml.yaml_utils import load_yaml
from pcd_processor import ExtractPointWorldIntoFrame, GetCAVLocation
from rainbow import MulticolorPatch

INPUT_OPV2V_PCD_PATH = None
OUTPUT_DIR = None
INPUT_DIRECTORY_GT_TRAJ = r'preprocessed_data/opv2v/gt_multiego_speedless/test/'

COLOR_MAP_CACHE = []
colors = ['blue', 'cyan', 'magenta', 'purple', 'orange', 'brown']

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--prediction_model_name', type=str, default=None, help='prediction mode name')
    parser.add_argument('--output_dir', type=str, default=None, help='folder to save outputs')
    parser.add_argument('--dataset', type=str, default=None, help='dataset_name')
    parser.add_argument('--dataset_path', type=str, default=None, help='dataset path')
    
    args = parser.parse_args()

    return args
    
def GetPointCloudWorldFrame(scenario_name: str,
                            timestamp_key: str,
                            ego_cav_id: str):
    path_to_pcd = Path(INPUT_OPV2V_PCD_PATH) / Path(scenario_name) / Path(ego_cav_id) / Path(f'{timestamp_key}.pcd')
    path_to_yaml = Path(INPUT_OPV2V_PCD_PATH) / Path(scenario_name) / Path(ego_cav_id) / Path(f'{timestamp_key}.yaml')
    if os.path.exists(path_to_pcd) and os.path.exists(path_to_yaml):
        return ExtractPointWorldIntoFrame(str(path_to_pcd), str(path_to_yaml))
    else:
        return None, None

def GetCAVLidarLocation(scenario_name: str,
                    timestamp_key: str,
                    cav_id: str):
    path_to_yaml = Path(INPUT_OPV2V_PCD_PATH) / Path(scenario_name) / Path(cav_id) / Path(f'{timestamp_key}.yaml')
    if os.path.exists(path_to_yaml):
        return GetCAVLocation(str(path_to_yaml))
    else:
        return None

def ScanOrLoadInputFiles(dir: str):
    variant = dir.split('/')[2]
    result_database_cache_path = f'Plotter/result_database_{variant}_cache.pickle'

    if os.path.exists(result_database_cache_path):
        print('Loading cache.')
        with open(result_database_cache_path, 'rb') as file:
            sorted_result = pickle.load(file)
        return sorted_result

    result_database = OrderedDict()

    # Get a list of files.
    all_files = []
    for file in glob.glob(dir):
        all_files.append(file)
    print(f"Detected {len(all_files)} files for {dir}.")

    # Read the files.
    for file in tqdm(all_files):
        with open(file, 'rb') as f:
            try:
                content = pickle.load(f)
            except:
                print(f'WARN: {file}')
                continue

            scenario_name = content['scenario_name']
            timestamp_idx = content['timestamp_idx']
            timestamp_key = content['timestamp_key']
            ego_cav_id = content['ego_cav_id']
            pred_trajs = content['pred_trajs']  # (6, 50, 2)
            gt_trajs = content['gt_trajs']  # (61, 8)
            distances_min_val = content['distances_min_val']
            distances_min_idx = content['distances_min_idx']
            final_distances_min_val = content['final_distances_min_val']
            final_distances_min_idx = content['final_distances_min_idx']

            # if content['center_gt_final_valid_idx'] < 35:
            #     continue
            #
            # if np.linalg.norm(gt_trajs[-1,:2] - gt_trajs[10,:2]) < 4:
            #     continue

            instance_key = (scenario_name, timestamp_key, ego_cav_id)

            if instance_key not in result_database:
                result_database[(scenario_name, timestamp_key, ego_cav_id)] = []
            result_database[(scenario_name, timestamp_key, ego_cav_id)].append({
                'object_id': content['object_id'],
                'distances_min_val' : distances_min_val,
                'best_pred_traj' : pred_trajs[distances_min_idx],
                'gt_trajs' : gt_trajs,
                'center_gt_trajs_mask' : content['center_gt_trajs_mask'],
            })

    with open(result_database_cache_path, 'wb') as file:
        pickle.dump(result_database, file)

    return result_database

def PlotObjects(scenario_name: str,
                timestamp_key: str,
                ego_cav_id: str,
                objects,
                variant: str,
                # ade: float,
                car_loc_allowed,
                only_ego_pcd: bool):
                
    x_lim = []
    y_lim = []

    # Plot the CAVs in this scenario.
    for cav_id in cavs_where_when[(scenario_name, timestamp_key)]:
        location = GetCAVLidarLocation(scenario_name, timestamp_key, cav_id)
        x, y, l, w, theta = location[0], location[1], 2.2, 1.0, location[4]

        if cav_id == ego_cav_id:
            rect = patches.Rectangle((x+l, y+w), 2 * l, 2 * w, linewidth=1, edgecolor='red', facecolor='red',
                                     angle=theta)
                                     
            x_lim = [x-100, x+100]
            y_lim = [y-50, y+50]
            points, intensities = GetPointCloudWorldFrame(scenario_name, timestamp_key, cav_id)
            plt.scatter(points[:, 0], points[:, 1], c=intensities, s=0.01)
        else:
            rect = patches.Rectangle((x+l, y+w), 2 * l, 2 * w, linewidth=1, edgecolor='black', facecolor='black',
                                     angle=theta)
            if not only_ego_pcd:
                points, intensities = GetPointCloudWorldFrame(scenario_name, timestamp_key, cav_id)
                plt.scatter(points[:, 0], points[:, 1], c=intensities, s=0.01)


        plt.gca().add_patch(rect)

    # Create a proxy artist for the legend
    legend_proxies = [Line2D([0], [0], linestyle="none", marker="s", markersize=4, markerfacecolor='red',
                             markeredgecolor='red')]
    legend_labels = ['Ego CAV']

    legend_proxies += [Line2D([0], [0], linestyle="none", marker="s", markersize=4, markerfacecolor='black',
                             markeredgecolor='black')]
    legend_labels += ['Other CAVs']

    # Plot top 6 objects.
    for k in range(len(objects)):
        object_id = str(objects[k]['object_id'])
        ade = objects[k]['distances_min_val']
        curr_loc = objects[k]['gt_trajs'][10, :2]
   
        # Sticky color assignment based on GT position.
        color = colors[k % len(colors)]  # Select color based on index
        found = False
        for (tgt_loc, tgt_color) in COLOR_MAP_CACHE:
            if np.linalg.norm(tgt_loc - curr_loc) < 0.5:
                color = tgt_color
                found = True
                break
        if not found:
            COLOR_MAP_CACHE.append((curr_loc, color))

        gt_traj = objects[k]['gt_trajs'][10]
        x, y, l, w, theta = gt_traj[0], gt_traj[1], gt_traj[3], gt_traj[4], np.rad2deg(gt_traj[6])

        rect = patches.Rectangle((x + l, y + w), 2 * l, 2 * w, linewidth=1, edgecolor=color, facecolor='none',
                                 angle=theta)

        plt.gca().add_patch(rect)

        # Plot trajectories.
        traj = objects[k]['best_pred_traj']
        traj = np.concatenate((objects[k]['gt_trajs'][:10, :2], traj), axis=0)
        plt.plot(traj[::4, 0], traj[::4, 1], 'x', markersize=2, color=color)

        # Plot gt.
        gt_traj = objects[k]['gt_trajs'][10:]
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], '-', markersize=0.4, linewidth=0.4, color='black')
        
    legend_proxies.append(MulticolorPatch(colors))
    legend_labels.append(f'Non-CAV Vehicles')

    # Create a proxy artist for predictions https://stackoverflow.com/questions/31478077/how-to-make-two-markers-share-the-same-label-in-the-legend
    proxy_prediction_1 = Line2D([0], [0], linestyle='None', marker='x', markersize=5, color='cyan')
    proxy_prediction_2 = Line2D([0], [0], linestyle='None', marker='x', markersize=5, color='brown')
    proxy_prediction_3 = Line2D([0], [0], linestyle='None', marker='x', markersize=5, color='purple')
    legend_proxies.append((proxy_prediction_1,proxy_prediction_2,proxy_prediction_3))
    legend_labels.append('Predicted Waypoints')

    # Create a proxy artist for ground truth
    proxy_ground_truth = Line2D([0], [0], linestyle='-', linewidth=1, color='black')
    legend_proxies.append(proxy_ground_truth)
    legend_labels.append('Ground Truth Trajectory')

    plt.gca().set_aspect('equal', 'box')
    
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])

    plt.axis('off')
    plt.show()
    flag = 'N' if only_ego_pcd else 'Y'
    plt.savefig(f'{OUTPUT_DIR}/{flag}_{scenario_name}_{timestamp_key}_{ego_cav_id}.png', bbox_inches='tight', dpi=300)

def PlotGTObjects(scenario_name: str, timestamp_key: str, ego_cav_id: str, variant: str, only_ego_pcd: bool):

    x_lim = []
    y_lim = []

    # Plot the CAVs in this scenario.
    other_cavs = []
    for cav_id in all_cavs_in_scenarios[scenario_name]:
        location = GetCAVLidarLocation(scenario_name, timestamp_key, cav_id)
        if location is None:
            continue
        x, y, l, w, theta = location[0], location[1], 2.2, 1.0, location[4]

        # 2021_08_18_19_48_05
        if cav_id == ego_cav_id:
            rect = patches.Rectangle((x+l, y+w), 2 * l, 2 * w, linewidth=1, edgecolor='red', facecolor='red',
                                     angle=theta)
                                     
            x_lim = [x-100, x+100]
            y_lim = [y-50, y+50]
            points, intensities = GetPointCloudWorldFrame(scenario_name, timestamp_key, cav_id)
            if points is None:
                continue
            plt.scatter(points[:, 0], points[:, 1], c=intensities, s=0.01)
        else:
            other_cavs.append(cav_id)
            rect = patches.Rectangle((x+l, y+w), 2 * l, 2 * w, linewidth=1, edgecolor='black', facecolor='black',
                                     angle=theta)
            if not only_ego_pcd:
                points, intensities = GetPointCloudWorldFrame(scenario_name, timestamp_key, cav_id)
                if points is None:
                    continue
                plt.scatter(points[:, 0], points[:, 1], c=intensities, s=0.01)

        plt.gca().add_patch(rect)

    # Create a proxy artist for the legend
    legend_proxies = [Line2D([0], [0], linestyle="none", marker="s", markersize=4, markerfacecolor='red',
                             markeredgecolor='red')]
    legend_labels = ['Ego CAV']

    legend_proxies += [Line2D([0], [0], linestyle="none", marker="s", markersize=4, markerfacecolor='black',
                             markeredgecolor='black')]
    legend_labels += ['Other CAVs']

    # Plot objects
    cur_params = load_yaml(os.path.join(INPUT_OPV2V_PCD_PATH, scenario_name, ego_cav_id, timestamp_key + '.yaml'))
    gt_trajs = {}
    timestamps = []
    with open(os.path.join(INPUT_DIRECTORY_GT_TRAJ, f'{scenario_name}-{ego_cav_id}-traj.pickle'), 'rb') as f:
        pkl = pickle.load(f)
        gt_trajs = pkl['data']
        timestamps = pkl['timestamps']
        
    idx = 0
    for vehicle_id, vehicle_params in cur_params['vehicles'].items():
        
        if vehicle_id in other_cavs:
            continue
        roll, theta, pitch = vehicle_params['angle']
        xyz = [loc + cen for loc, cen in zip(vehicle_params['location'], vehicle_params['center'])]
        l, w, h = vehicle_params['extent']

        color = colors[idx % len(colors)]  # Select color based on index

        if timestamp_key not in timestamps:
            continue
        current_t = timestamps.index(timestamp_key)
        if current_t + 50 > len(gt_trajs[vehicle_id]):
            traj = gt_trajs[vehicle_id][current_t:]
        else:
            traj = gt_trajs[vehicle_id][current_t:current_t+50]
        traj = np.array(traj)

        is_valid = True
        for position in traj:
            if position[-1] == 0:
                is_valid = False
        if not is_valid:
            continue

        # Plot rectangles
        rect = patches.Rectangle((xyz[0] + l, xyz[1] + w), 2 * l, 2 * w, linewidth=1, edgecolor=color, facecolor='none', angle=theta)
        idx += 1
        # Plot trajectories
        plt.gca().add_patch(rect)
        plt.plot(traj[:, 0], traj[:, 1], '-', markersize=0.4, linewidth=0.4, color='black')
    
    plt.gca().set_aspect('equal', 'box')
    plt.xlim(x_lim[0], x_lim[1])
    plt.ylim(y_lim[0], y_lim[1])
    
    plt.axis('off')
    plt.show()
    flag = 'N' if only_ego_pcd else 'Y'
    plt.savefig(f'{OUTPUT_DIR}/{flag}_{scenario_name}_{timestamp_key}_{ego_cav_id}.png', bbox_inches='tight', dpi=300)
    
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

if __name__ == "__main__":

    args = parse_config()

    prediction_model_name = args.prediction_model_name
    dataset = args.dataset
    INPUT_OPV2V_PCD_PATH = args.dataset_path
    if args.output_dir is None:
        OUTPUT_DIR = f'Plotter/visualizations/{dataset}-{prediction_model_name}'
    else:
        OUTPUT_DIR = args.output_dir
        
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    if prediction_model_name == 'cmp':
        prediction_model_name = f'{dataset}_multiego_cobevt_c256'
    elif prediction_model_name == 'no_coop':
        prediction_model_name = f'{dataset}_multiego_no_coop'
    elif prediction_model_name == 'v2vnet':
        prediction_model_name = f'{dataset}_multiego_v2vnet'
    elif prediction_model_name == 'cooperative_perception_only':
        prediction_model_name = f'{dataset}_multiego_cobevt_c256_no_agg'

    inference_results_path = os.path.join(f'MTR/output/{prediction_model_name}/default/eval', 'inference_result/*.pkl')

    results = ScanOrLoadInputFiles(inference_results_path)

    # Find all CAVs in a scenario and timestamp.
    cavs_where_when = {}
    timestamp_keys = {}
    all_cavs_in_scenarios = {}

    scenarios = os.listdir(os.path.join(INPUT_OPV2V_PCD_PATH))
    for scenario in scenarios:
        all_cavs_in_scenarios[scenario] = [x for x in os.listdir(os.path.join(INPUT_OPV2V_PCD_PATH, scenario)) if os.path.isdir(os.path.join(INPUT_OPV2V_PCD_PATH, scenario, x))]
        
    for (scenario_name, timestamp_key, ego_cav_id) in results:
        if (scenario_name, timestamp_key) not in cavs_where_when:
            cavs_where_when[(scenario_name, timestamp_key)] = set()
        if (scenario_name, ego_cav_id) not in timestamp_keys:
            timestamp_keys[(scenario_name, ego_cav_id)] = set()
        cavs_where_when[(scenario_name, timestamp_key)].add(ego_cav_id)
        timestamp_keys[(scenario_name, ego_cav_id)].add(timestamp_key)

    sorted_results = sorted(results, key=lambda x: (x[0], x[1], x[2]))
    # sort timestamp_keys's key in ascending order
    timestamp_keys = OrderedDict(sorted(timestamp_keys.items(), key=lambda x: (x[0][0], x[0][1])))
    # sort timestamp_keys's value in ascending order
    for (scenario_name, ego_cav_id) in timestamp_keys:
        timestamp_keys[(scenario_name, ego_cav_id)] = sorted(list(timestamp_keys[(scenario_name, ego_cav_id)]))

    # Plot
    legend_proxies = []
    legend_labels = []

    for (scenario_name, ego_cav_id) in tqdm(timestamp_keys):
        
        all_timestamps = extract_timestamps([f for f in os.listdir(os.path.join(INPUT_OPV2V_PCD_PATH, scenario_name, ego_cav_id)) if f.endswith('.yaml')])
        
        for timestamp_key in all_timestamps:
            print(f'Plotting at scenario {scenario_name}, timestamp {timestamp_key}, ego_cav_id {ego_cav_id}.')

            COLOR_MAP_CACHE.clear()

            if (scenario_name, timestamp_key, ego_cav_id) in sorted_results:
                car_loc = [x['gt_trajs'][10, :2] for x in results[(scenario_name, timestamp_key, ego_cav_id)]]
                if prediction_model_name == f'{dataset}_multiego_no_coop':
                    PlotObjects(scenario_name, timestamp_key, ego_cav_id, results[(scenario_name, timestamp_key, ego_cav_id)], prediction_model_name, car_loc, True)
                else:
                    PlotObjects(scenario_name, timestamp_key, ego_cav_id, results[(scenario_name, timestamp_key, ego_cav_id)], prediction_model_name, car_loc, False)
                plt.clf()
            else:
                if prediction_model_name == f'{dataset}_multiego_no_coop':
                    PlotGTObjects(scenario_name, timestamp_key, ego_cav_id, prediction_model_name, True)
                else:
                    PlotGTObjects(scenario_name, timestamp_key, ego_cav_id, prediction_model_name, False)
                plt.clf()