import gc
import os
import pickle
import sys
import time

import numpy as np
import torch
import tqdm
import psutil
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

from mtr.utils import common_utils
from pympler import asizeof

from mtr.datasets.opv2v_multiego_dataset import OPV2VMultiEgoDataset
from mtr.datasets.v2v4real_multiego_dataset import V2V4RealMultiEgoDataset
from mtr.datasets.waymo.waymo_eval import waymo_evaluation, transform_preds_to_waymo_format

def fit_polynomial_and_evaluate_errors(trajectory, order=2, threshold=0.1):
    # Separate x and y coordinates
    x, y = trajectory[:, 0], trajectory[:, 1]

    # Fit a polynomial
    p = Polynomial.fit(x, y, order)

    # Predict and calculate errors
    y_pred = p(x)
    errors = np.abs(y - y_pred)

    # Visualize
    # x_line = np.linspace(min(x), max(x), 100)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # ax.scatter(x, y, color='blue', label='Data Points')
    # ax.plot(x_line, p(x_line), color='red', label='Polynomial Fit')
    # ax.legend()
    # plt.savefig(os.path.join('Visualization', f"fit.png"))
    # plt.close(fig)

    # Evaluate errors
    history_errors = errors[:10]
    future_errors = errors[10:]

    return np.any(future_errors > threshold), history_errors, future_errors

def eval_one_epoch_custom(model, dataloader, epoch_id, logger,
                          dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    """
        Ref: https://arxiv.org/pdf/2104.10133.pdf
             https://github.com/waymo-research/waymo-open-dataset/blob/36529372c6aec153e96ae8e928deacfdf7479c5f/src/waymo_open_dataset/metrics/motion_metrics.h
    """
    # Allocate save file.
    if save_to_file:
        result_dir.mkdir(parents=True, exist_ok=True)
        final_output_dir = result_dir / 'inference_result'
        final_output_dir.mkdir(parents=True, exist_ok=True)


    logger.info('*************** EPOCH %s CUSTOM EVALUATION *****************' % epoch_id)
    # Get all results from the model.
    trajectories_count = 0
    total_distance_error = 0
    total_distance_error_1s = 0
    total_distance_error_3s = 0
    total_final_distance_error = 0
    total_final_distance_error_1s = 0
    total_final_distance_error_3s = 0
    total_miss_count_3_6 = 0
    total_miss_count_3 = 0
    total_miss_count_2 = 0
    total_miss_count_1_8 = 0
    anomal_gt_counter = 0

    model.eval()
    for batch_dict in tqdm.tqdm(dataloader):
        with torch.no_grad():

            batch_pred_dict = model(batch_dict)
            pred_dicts_cavs_list = OPV2VMultiEgoDataset.generate_prediction_dicts(batch_pred_dict, output_path=final_output_dir if save_to_file else None)
            # pred_dicts_cavs_list = V2V4RealMultiEgoDataset.generate_prediction_dicts(batch_pred_dict, output_path=final_output_dir if save_to_file else None)

            for pred_dicts_cav in pred_dicts_cavs_list:
                
                # Compute ADE, FDE, MR and mAP.
                for pred_dict in pred_dicts_cav:
                    scenario_name = pred_dict['scenario_name']
                    timestamp_idx = pred_dict['timestamp_idx']
                    obj_id = pred_dict['object_id']
                    ego_cav_id = pred_dict['ego_cav_id']

                    gt_trajs = pred_dict['gt_trajs']
                    gt_trajs_future = gt_trajs[-50:]
                    pred_trajs = pred_dict['pred_trajs']
                    pred_scores = pred_dict['pred_scores']
                    center_gt_trajs_mask = pred_dict['center_gt_trajs_mask']
                    center_gt_final_valid_idx = pred_dict['center_gt_final_valid_idx']

                    # Filter GT.
                    # is_anomaly, history_errors, future_errors = fit_polynomial_and_evaluate_errors(gt_trajs, order=3, threshold=0.18)

                    # if (is_anomaly):
                    #     anomal_gt_counter += 1
                    #     continue
                    # else:
                    #     # continue                        

                    # print("Anomaly in Future Data:", is_anomaly)
                    # print("History Errors:", history_errors)
                    # print("Future Errors:", future_errors)

                
                    # Compute 5s ADE.
                    distances_modes_timesteps = np.linalg.norm(pred_trajs[:, :center_gt_final_valid_idx, 0:2] - gt_trajs_future[:center_gt_final_valid_idx, 0:2], axis=-1) # (num_modes, num_timesteps)
                    distances_modes_timesteps = distances_modes_timesteps
                    distances_modes = distances_modes_timesteps.mean(axis=-1)  # (num_modes)
                    distances_min_val = distances_modes.min()  # 1
                    # if np.isnan(distances_min_val):
                    #     continue
                    distances_min_idx = distances_modes.argmin()  # 1

                    # Compute 3s ADE.
                    distances_modes_timesteps_3s = np.linalg.norm(pred_trajs[:, :min(30, center_gt_final_valid_idx), 0:2] - gt_trajs_future[:min(30, center_gt_final_valid_idx), 0:2], axis=-1) # (num_modes, num_timesteps)
                    distances_modes_timesteps_3s = distances_modes_timesteps_3s
                    distances_modes_3s = distances_modes_timesteps_3s.mean(axis=-1)  # (num_modes)
                    distances_min_val_3s = distances_modes_3s.min()  # 1

                    # Compute 1s ADE.
                    distances_modes_timesteps_1s = np.linalg.norm(pred_trajs[:, :min(10, center_gt_final_valid_idx), 0:2] - gt_trajs_future[:min(10, center_gt_final_valid_idx), 0:2], axis=-1) # (num_modes, num_timesteps)
                    distances_modes_timesteps_1s = distances_modes_timesteps_1s
                    distances_modes_1s = distances_modes_timesteps_1s.mean(axis=-1)  # (num_modes)
                    distances_min_val_1s = distances_modes_1s.min()  # 1

                    # Compute FDE.
                    final_distances_modes = np.linalg.norm(pred_trajs[:, center_gt_final_valid_idx, 0:2] - gt_trajs_future[center_gt_final_valid_idx, 0:2], axis=-1) # (num_modes)
                    final_distances_min_val = final_distances_modes.min()  # 1
                    # if np.isnan(final_distances_min_val):
                    #     continue
                    final_distances_min_idx = final_distances_modes.argmin()  # 1

                    # Compute 3s FDE.
                    final_distances_modes_3s = np.linalg.norm(pred_trajs[:, min(30, center_gt_final_valid_idx), 0:2] - gt_trajs_future[min(30, center_gt_final_valid_idx), 0:2], axis=-1) # (num_modes)
                    final_distances_min_val_3s = final_distances_modes_3s.min()  # 1

                    # Compute 1s FDE.
                    final_distances_modes_1s = np.linalg.norm(pred_trajs[:, min(10, center_gt_final_valid_idx), 0:2] - gt_trajs_future[min(10, center_gt_final_valid_idx), 0:2], axis=-1) # (num_modes)
                    final_distances_min_val_1s = final_distances_modes_1s.min()  # 1
                    
                    # print(f'ADE {distances_min_val}, FDE {final_distances_min_val}')
                    # fig = plt.figure(figsize=(10, 10))
                    # ax = fig.add_subplot(111)
                    # ax.set_xlabel('x')
                    # ax.set_ylabel('y')
                    # ax.scatter(gt_trajs[:, 0], gt_trajs[:, 1], color='k', label='GT')
                    # # ax.scatter(pred_trajs[distances_min_idx, :, 0], pred_trajs[distances_min_idx, :, 1], color='r', label='Pred')
                    # for l in range(6):
                    #     ax.scatter(pred_trajs[l, :, 0], pred_trajs[l, :, 1], label='Pred'+str(l))
                    # # ax.axis('square')
                    # ax.legend()
                    # plt.savefig(os.path.join('Visualization', f"curr.png"))
                    # plt.close(fig)
                    # exit(0)

                    # Compute Miss Count. (Simplified such that we do not consider heading or speed.)
                    total_miss_count_3_6 += 1 if final_distances_min_val > 3.6 else 0
                    total_miss_count_3   += 1 if final_distances_min_val > 3.0 else 0
                    total_miss_count_2   += 1 if final_distances_min_val > 2.0 else 0
                    total_miss_count_1_8 += 1 if final_distances_min_val > 1.8 else 0

                    # Compute mAP. TODO
                    total_distance_error += distances_min_val
                    total_distance_error_3s += distances_min_val_3s
                    total_distance_error_1s += distances_min_val_1s
                    total_final_distance_error += final_distances_min_val
                    total_final_distance_error_3s += final_distances_min_val_3s
                    total_final_distance_error_1s += final_distances_min_val_1s
                    trajectories_count += 1

                    pred_dict['distances_min_val'] = distances_min_val
                    pred_dict['distances_min_val_3s'] = distances_min_val_3s
                    pred_dict['distances_min_val_1s'] = distances_min_val_1s
                    pred_dict['distances_min_idx'] = distances_min_idx
                    pred_dict['final_distances_min_val'] = final_distances_min_val
                    pred_dict['final_distances_min_val_3s'] = final_distances_min_val_3s
                    pred_dict['final_distances_min_val_1s'] = final_distances_min_val_1s
                    pred_dict['final_distances_min_idx'] = final_distances_min_idx

                    # Save results to file.
                    if save_to_file and os.listdir(final_output_dir).__len__() < 100000:
                        title = f"{scenario_name}-{ego_cav_id}-{timestamp_idx}--{obj_id}"

                        # Collect into a dict.
                        with open(os.path.join(final_output_dir, f'{title}.pkl'), 'wb') as f:
                            pickle.dump(pred_dict, f)

                        # Save the plot.
                        # fig = plt.figure(figsize=(10, 10))
                        # ax = fig.add_subplot(111)
                        # ax.set_xlabel('x')
                        # ax.set_ylabel('y')
                        # ax.set_title('Trajectories %s' % title)
                        # ax.scatter(gt_trajs[:10, 0], gt_trajs[:10, 1], color='k', label='Tracked History')
                        # ax.scatter(gt_trajs_future[:, 0], gt_trajs_future[:, 1], color='g', label='GT')
                        # ax.scatter(pred_trajs[distances_min_idx, :, 0], pred_trajs[distances_min_idx, :, 1], color='r', label='Pred')
                        # # ax.plot(pred_trajs[final_distances_min_idx, :, 0], pred_trajs[final_distances_min_idx, :, 1], 'b', label='Pred')
                        # ax.axis('square')
                        # ax.legend()
                        # plt.savefig(os.path.join(final_output_dir, f"{title}.png"))
                        # plt.close(fig)
        # break

    # Show results.
    logger.info('Results ------------------------------------------------')
    logger.info('       minADE 1s: %.4f' % (total_distance_error_1s / trajectories_count))
    logger.info('       minADE 3s: %.4f' % (total_distance_error_3s / trajectories_count))
    logger.info('       minADE 5s: %.4f' % (total_distance_error / trajectories_count))
    logger.info('       minFDE 1s: %.4f' % (total_final_distance_error_1s / trajectories_count))
    logger.info('       minFDE 3s: %.4f' % (total_final_distance_error_3s / trajectories_count))
    logger.info('       minFDE 5s: %.4f' % (total_final_distance_error / trajectories_count))
    logger.info('       MR 3.6: %.4f' % (total_miss_count_3_6 / trajectories_count))
    logger.info('       MR 3.0: %.4f' % (total_miss_count_3 / trajectories_count))
    logger.info('       MR 2.0: %.4f' % (total_miss_count_2 / trajectories_count))
    logger.info('       MR 1.8: %.4f' % (total_miss_count_1_8 / trajectories_count))
    logger.info('       Filtered: %d' % (anomal_gt_counter))
    logger.info('       Total: %d' % (trajectories_count))

    logger.info('****************CUSTOM Evaluation done.*****************')

    return {'minADE': total_distance_error / trajectories_count,
            'minFDE': total_final_distance_error / trajectories_count,
            'MR': total_miss_count_3_6 / trajectories_count}  # tensorboard

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            batch_pred_dicts = model(batch_dict)
            if cfg.DATA_CONFIG.DATASET == 'OPV2VMultiEgoDataset':
                final_pred_dicts = OPV2VMultiEgoDataset.generate_prediction_dicts(batch_pred_dicts,
                                                                              output_path=final_output_dir if save_to_file else None)
            elif cfg.DATA_CONFIG.DATASET == 'OPV2VDataset':
                final_pred_dicts = OPV2VDataset.generate_prediction_dicts(batch_pred_dicts,
                                                                              output_path=final_output_dir if save_to_file else None)

            del batch_pred_dicts
            pred_dicts += final_pred_dicts

        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}'
                        f'result_size: {asizeof.asizeof(pred_dicts) / 1e6} MB, '
                        f'ram_percent {psutil.virtual_memory()[2]}%')
            if psutil.virtual_memory()[2] > 90:
                logger.warn('Stopping eval due to low memory.')
                break
            gc.collect()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
        if pred_dicts is not None:
            logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir, 
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass
