# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])

    # Calculate Average Recall
    ar = sum(mrec) / len(mrec)
    return ap, ar, mrec, mpre

def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, gt_object_id_tensor=None):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    matched_indices = []
    matched_gt_indeces = set()
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            ious = np.array([iou if idx not in matched_gt_indeces else 0 for idx, iou in enumerate(ious)])

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                if gt_object_id_tensor is not None:
                    matched_indices.append(-1)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            if gt_object_id_tensor is not None:
                matched_indices.append(gt_object_id_tensor[gt_index].item())
            matched_gt_indeces.add(gt_index)
            # gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt
    if gt_object_id_tensor is not None:
        result_stat[iou_thresh]['matched_indices'] += matched_indices


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
        
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']
    if gt_total == 0:
        return 0,0,0,0

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, ar, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, ar, mrec, mprec

def get_f1_score(presicion, recall):
    if presicion + recall == 0:
        return 0
    return 2 * (presicion * recall) / (presicion + recall)

def eval_final_results(result_stat, save_path=None, global_sort_detections=False):
    dump_dict = {}

    ap_30, ar_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30, global_sort_detections)
    ap_50, ar_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50, global_sort_detections)
    ap_70, ar_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70, global_sort_detections)

    f1_30, f1_50, f1_70 = get_f1_score(ap_30, ar_30), get_f1_score(ap_50, ar_50), get_f1_score(ap_70, ar_70)

    dump_dict.update({'ap_30': ap_30,
                      'ar_30': ar_30,
                      'f1_30': f1_30,
                      'ap_50': ap_50,
                      'ar_50': ar_50,
                      'f1_50': f1_50,
                      'ap_70': ap_70,
                      'ar_70': ar_70,
                      'f1_70': f1_70,
                      'mpre_30': mpre_30,
                      'mrec_30': mrec_30,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })

    print('The Average Precision at IOU 0.3 is %.2f, the Average Recall at IOU 0.3 is %.2f, the F1 Score at IOU 0.3 is %.2f\n'
        'The Average Precision at IOU 0.5 is %.2f, the Average Recall at IOU 0.5 is %.2f, the F1 Score at IOU 0.5 is %.2f\n'
        'The Average Precision at IOU 0.7 is %.2f, the Average Recall at IOU 0.7 is %.2f, the F1 Score at IOU 0.7 is %.2f' % (ap_30, ar_30, f1_30, ap_50, ar_50, f1_50, ap_70, ar_70, f1_70))

    if save_path:
        output_file = 'eval.yaml' if not global_sort_detections else 'eval_global_sort.yaml'
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))


