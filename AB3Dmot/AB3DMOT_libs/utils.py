# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import yaml, numpy as np, os
from easydict import EasyDict as edict
# from AB3DMOT_libs.model_multi import AB3DMOT_multi
from AB3DMOT_libs.model import AB3DMOT
from AB3DMOT_libs.kitti_oxts import load_oxts
from AB3DMOT_libs.kitti_calib import Calibration
from AB3DMOT_libs.nuScenes_split import get_split
from AB3DMOT_libs.io import load_6dof_oxts
from xinshuo_io import mkdir_if_missing, is_path_exists, fileparts, load_list_from_folder
from xinshuo_miscellaneous import merge_listoflist

def Config(filename):
    listfile1 = open(filename, 'r')
    listfile2 = open(filename, 'r')
    cfg = edict(yaml.safe_load(listfile1))
    settings_show = listfile2.read().splitlines()

    listfile1.close()
    listfile2.close()

    return cfg, settings_show

def get_subfolder_seq(dataset, split):

	# dataset setting
	file_path = os.path.dirname(os.path.realpath(__file__))
	if dataset == 'KITTI':				# KITTI
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
		
		if split == 'val': subfolder = 'training' 
		elif split == 'test': subfolder = 'testing' 
		else: assert False, 'error'

		hw = {'image': (375, 1242), 'lidar': (720, 1920)}
		
		if split == 'train': seq_eval = ['0000', '0002', '0003', '0004', '0005', '0007', '0009', '0011', '0017', '0020']         # train
		# if split == 'val':   seq_eval = ['0001', '0006', '0008', '0010', '0012', '0013', '0014', '0015', '0016', '0018', '0019']    # val
		if split == 'val':   seq_eval = ['0001', '0016']
		if split == 'test':  seq_eval  = ['%04d' % i for i in range(29)]
	
		data_root = os.path.join(file_path, '../data/KITTI') 		# path containing the KITTI root 
	
	elif dataset == 'OPV2V':
		det_id2str = {2: 'Car'}
		if split == 'test': seq_eval = ['2021_08_18_19_48_05', '2021_08_20_21_10_24', '2021_08_21_09_28_12', '2021_08_22_07_52_02',
		'2021_08_22_09_08_29', '2021_08_22_21_41_24', '2021_08_23_12_58_19', '2021_08_23_15_19_19', '2021_08_23_16_06_26',
		'2021_08_23_17_22_47', '2021_08_23_21_07_10', '2021_08_23_21_47_19', '2021_08_24_07_45_41', '2021_08_24_11_37_54',
		'2021_08_24_20_09_18', '2021_08_24_20_49_54']
		elif split == 'train': seq_eval = ['2021_08_18_23_23_19', '2021_08_22_06_43_37', '2021_09_09_23_21_21', '2021_08_23_22_31_01', 
		'2021_08_23_19_27_57', '2021_08_24_12_19_30', '2021_08_22_10_46_58', '2021_08_23_11_06_41', '2021_08_24_09_58_32', '2021_08_23_17_07_55', 
		'2021_08_16_22_26_54', '2021_08_18_18_33_56', '2021_08_21_15_41_04', '2021_08_23_16_42_39', '2021_09_10_12_07_11', '2021_09_09_22_21_11', 
		'2021_08_21_21_35_56', '2021_08_22_09_43_53', '2021_08_22_10_10_40', '2021_08_18_19_11_02', '2021_08_22_08_39_02', '2021_08_18_21_38_28', 
		'2021_08_23_23_08_17', '2021_08_23_20_47_11', '2021_08_23_11_22_46', '2021_08_20_16_20_46', '2021_08_21_09_09_41', '2021_08_20_21_00_19', 
		'2021_08_23_10_47_16', '2021_08_24_09_25_42', '2021_08_22_07_24_12', '2021_08_24_21_29_28', '2021_08_21_16_08_42', 
		'2021_08_23_13_10_47', '2021_08_21_17_00_32', '2021_08_22_22_30_58', '2021_08_20_20_39_00', '2021_08_18_09_02_56', '2021_08_19_15_07_39', 
		'2021_08_22_11_29_38', '2021_08_23_12_13_48', '2021_08_21_22_21_37', '2021_08_18_22_16_12'] #'2021_09_09_13_20_58' has not bev lane images
		elif split == 'train_mini': seq_eval = ['2021_08_16_22_26_54']
		subfolder = '' 
		hw = {'image': (375, 1242), 'lidar': (720, 1920)} #not used
		data_root = os.path.join(file_path, '../data/OPV2V')
	
	elif dataset == 'nuScenes':			# nuScenes
		det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Bicycle', 4: 'Motorcycle', 5: 'Bus', \
			6: 'Trailer', 7: 'Truck', 8: 'Construction_vehicle', 9: 'Barrier', 10: 'Traffic_cone'}

		subfolder = split
		hw = {'image': (900, 1600), 'lidar': (720, 1920)}

		if split == 'train': seq_eval = get_split()[0]		# 700 scenes
		if split == 'val':   seq_eval = get_split()[1]		# 150 scenes
		if split == 'test':  seq_eval = get_split()[2]      # 150 scenes

		data_root = os.path.join(file_path, '../data/nuScenes/nuKITTI') 	# path containing the nuScenes-converted KITTI root

	else: assert False, 'error, %s dataset is not supported' % dataset
		
	return subfolder, det_id2str, hw, seq_eval, data_root

def get_threshold(dataset, det_name):
	# used for visualization only as we want to remove some false positives, also can be 
	# used for KITTI 2D MOT evaluation which uses a single operating point 
	# obtained by observing the threshold achieving the highest MOTA on the validation set

	if dataset == 'KITTI':
		if det_name == 'pointrcnn': return {'Car': 3.240738, 'Pedestrian': 2.683133, 'Cyclist': 3.645319}
		else: assert False, 'error, detection method not supported for getting threshold' % det_name
	elif dataset == "OPV2V":
		if det_name == 'pointpillar-CoBEVT-nocompression': return {'Car': 0.25} #random setting
	elif dataset == 'nuScenes':
		if det_name == 'megvii': 
			return {'Car': 0.262545, 'Pedestrian': 0.217600, 'Truck': 0.294967, 'Trailer': 0.292775, 
					'Bus': 0.440060, 'Motorcycle': 0.314693, 'Bicycle': 0.284720}
		if det_name == 'centerpoint': 
			return {'Car': 0.269231, 'Pedestrian': 0.410000, 'Truck': 0.300000, 'Trailer': 0.372632, 
					'Bus': 0.430000, 'Motorcycle': 0.368667, 'Bicycle': 0.394146}
		else: assert False, 'error, detection method not supported for getting threshold' % det_name
	else: assert False, 'error, dataset %s not supported for getting threshold' % dataset

def pose_to_transformation_matrix(pose):
    """
    Parameters
    ----------
    pose: np.ndarray
        (N, 6), the pose of the object [x, y, z, roll, pitch, yaw].

    Returns:
    --------
    transformation_matrix: np.ndarray
        (N, 4, 4), the transformation matrix of the pose.
    """
    transformation_matrix = np.zeros((pose.shape[0], 4, 4))

    for i in range(pose.shape[0]):
        x, y, z, roll, pitch, yaw = pose[i]

        # Rotation matrix
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        R = np.dot(R_z, np.dot(R_y, R_x))

        # Translation vector
        T = np.array([x, y, z])

        # Transformation matrix
        transformation_matrix[i, :3, :3] = R
        transformation_matrix[i, :3, 3] = T
        transformation_matrix[i, 3, 3] = 1

    return transformation_matrix

def initialize(cfg, data_root, save_dir, subfolder, seq_name, cat, ID_start, hw, log_file):
	# initialize the tracker and provide all path of data needed

	oxts_dir  = os.path.join(data_root, subfolder, 'oxts')
	calib_dir = os.path.join(data_root, subfolder, 'calib')
	image_dir = os.path.join(data_root, subfolder, 'image_02')

	# load ego poses
	if cfg.dataset == 'OPV2V':
		poses = load_6dof_oxts(os.path.join(data_root, subfolder, 'oxts', seq_name+'.txt'))
		imu_poses = pose_to_transformation_matrix(poses)
	else:
		oxts = os.path.join(data_root, subfolder, 'oxts', seq_name+'.json')
		if not is_path_exists(oxts): oxts = os.path.join(data_root, subfolder, 'oxts', seq_name+'.txt')
		imu_poses = load_oxts(oxts) if is_path_exists(oxts) else None              # seq_frames x 4 x 4

	# load calibration
	if cfg.dataset == 'OPV2V':
		calib = None
	else:
		calib = os.path.join(data_root, subfolder, 'calib', seq_name+'.txt')
		calib = Calibration(calib)

	# load image for visualization
	if cfg.dataset == 'OPV2V':
		img_seq = None
		vis_dir = None
	else:
		img_seq = os.path.join(data_root, subfolder, 'image_02', seq_name)
		vis_dir = os.path.join(save_dir, 'vis_debug', seq_name); mkdir_if_missing(vis_dir)

	# initiate the tracker
	if cfg.num_hypo > 1:
		tracker = AB3DMOT_multi(cfg, cat, calib=calib, oxts=imu_poses, img_dir=img_seq, vis_dir=vis_dir, hw=hw, log=log_file, ID_init=ID_start) 
	elif cfg.num_hypo == 1:
		tracker = AB3DMOT(cfg, cat, calib=calib, oxts=imu_poses, img_dir=img_seq, vis_dir=vis_dir, hw=hw, log=log_file, ID_init=ID_start) 
	else: assert False, 'error'
	
	# compute the min/max frame
	if cfg.dataset == 'OPV2V':
		frame_list = list(str(i) for i in range(imu_poses.shape[0]))
	else:
		frame_list, _ = load_list_from_folder(img_seq)
		frame_list = [fileparts(frame_file)[1] for frame_file in frame_list]

	return tracker, frame_list

def find_all_frames(root_dir, subset, data_suffix, seq_list):
	# warm up to find union of all frames from results of all categories in all sequences
	# finding the union is important because there might be some sequences with only cars while
	# some other sequences only have pedestrians, so we may miss some results if mainly looking
	# at one single category
	# return a dictionary with each key correspondes to the list of frame ID

	# loop through every sequence
	frame_dict = dict()
	for seq_tmp in seq_list:
		frame_all = list()

		# find all frame indexes for each category
		for subset_tmp in subset:
			data_dir = os.path.join(root_dir, subset_tmp, 'trk_withid'+data_suffix, seq_tmp)			# pointrcnn_ped
			if not is_path_exists(data_dir):
				print('%s dir not exist' % data_dir)
				assert False, 'error'

			# extract frame string from this category
			frame_list, _ = load_list_from_folder(data_dir)
			frame_list = [fileparts(frame_tmp)[1] for frame_tmp in frame_list]
			frame_all.append(frame_list)
		
		# merge frame indexes from all categories
		frame_all = merge_listoflist(frame_all, unique=True)
		frame_dict[seq_tmp] = frame_all

	return frame_dict