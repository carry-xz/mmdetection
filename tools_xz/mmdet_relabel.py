import pandas as pd
import numpy as np
import os, urllib, cv2, sys, io
from PIL import Image, ImageFile
from tqdm import tqdm
import shutil

from mmdet.apis import init_detector, inference_detector
from xml_lib import Xml
from mmdet.datasets import build_dataset
from mmcv import Config
from mmcv.ops import nms
import torch
# help to label


def splistImg_(img, num_x=1, num_y=1, max_len=None, over_lap=0.015):
    '''
    :param img: ndarray (opencv imread)
    :param num_x: int
    :param num_y: int
    :param over_lap: float
    :return: list ndarray :imgs, box limits
    '''
    if max_len is not None:
        height, width = img.shape[:2]
        num_x = max(width // max_len, 1)
        num_y = max(height // max_len, 1)
    res_imgs = []
    limit_xxyys = []
    height, width = img.shape[:2]
    img_w, img_h = width // num_x + 1, height // num_y + 1
    over_x, over_y = int(img_w * over_lap), int(img_h * over_lap)
    for i in range(0, num_y):
        for j in range(0, num_x):
            x1 = max(j * img_w - over_x, 0)
            x2 = min(width, (j + 1) * img_w + over_x)
            y1 = max(i * img_h - over_y, 0)
            y2 = min(height, (i + 1) * img_h + over_y)
            img_ = img[y1:y2, x1:x2, :]
            res_imgs.append(img_)
            limit_xxyys.append([x1, x2, y1, y2])
            # cv2.imwrite('test_{}_{}.jpg'.format(j,i),img_)
            # print('\nlimit:',[x1, x2, y1, y2])
    return res_imgs, limit_xxyys


def cleanResByBoundary(xyxycs_all, limit_xxyy):
    '''
    clean bbox which torch the boundary
    :param xywhcs_all:
    :param limit_xxyy:
    :return:
    '''
    # print(xyxycs_all)
    res = []
    for xyxycs in xyxycs_all:
        if len(xyxycs) == 0:
            res.append(xyxycs)
            continue
        xyxys_ = xyxycs.copy()
        x_min, x2, y_min, y2 = limit_xxyy
        w, h = x2 - x_min, y2 - y_min
        mask_sel = (xyxys_[:, 0] <= 0) | (xyxys_[:, 1] <= 0) | (xyxys_[:, 2] >= w - 1) | (xyxys_[:, 3] >= h - 1)
        xywhs_sel = xyxys_[~mask_sel].copy()
        xywhs_sel[:, 0] += x_min
        xywhs_sel[:, 1] += y_min
        xywhs_sel[:, 2] += x_min
        xywhs_sel[:, 3] += y_min
        res.append(xywhs_sel)

    return res


def merageResults(res_det, limit_xxyys, skip_first=False):
    '''
    :param res_det: [img1_res,img2_res...] img1_res:[c1_res,c2_res,...]
    :param limit_xxyys: [[xxyy,xxyy,xxyy]]
    :return: xyxyc
    '''
    res_all = None
    for i, xyxycs in enumerate(res_det):
        if skip_first:
            skip_first = False
            res_all = xyxycs
        else:
            xyxycs_i = cleanResByBoundary(xyxycs, limit_xxyys[i])
            if res_all is None:
                res_all = xyxycs_i
                continue
            for i, item in enumerate(xyxycs_i):
                if len(item) == 0:
                    continue
                if len(res_all[i]) == 0:
                    res_all[i] = xyxycs_i[i]
                else:
                    res_all[i] = np.vstack((res_all[i], xyxycs_i[i]))
    return res_all


def model_init(config_file, checkpoint_file):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    return model


def detect(model, img, split=False, max_len=900):
    if not split:
        results_all = inference_detector(model, img)
    else:
        imgs, limits_xxyy = splistImg_(img, max_len=max_len)
        results_all = []
        if len(imgs) > 1:
            print('split img to {} part'.format(len(imgs)))
        for i, img_i in enumerate(imgs):
            result_ = inference_detector(model, img_i)
            results_all.append(result_)
            # img_ = plotResultOnImg(img_i,result_,0.3)
            # cv2.imwrite('aa{}.jpg'.format(i),img_)
        results_all = merageResults(results_all, limits_xxyy)
    return results_all


def plotResultOnImg(img, results, conf_th, bbox_color=(0, 255, 0), thickness=1):
    for result in results:
        bboxes = result[result[:, 4] >= conf_th]

        for bbox in bboxes:
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            text_color = bbox_color
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = 'y'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, text_color)
    return img


def getFileListFromDir(tar_dir, file_type=('jpg', 'png', 'JPG', 'PNG', 'Jpg', 'Png')):
    res = []
    for root, dirs, files in os.walk(tar_dir):
        for file in files:
            if file.endswith(file_type):
                res.append(os.path.join(root, file))
    print(tar_dir)
    print('file number:', len(res))
    return res


def getBBoxFromObjList(ann1_dic, name_list, prob=1.):
    if isinstance(name_list, tuple):
        name_list = list(name_list)
    gt_objs_list = ann1_dic['objs_list']
    res = [[] for i in range(len(name_list))]
    for item in gt_objs_list:
        class_name = item['name']
        if class_name in err_name_dic:
            class_name = err_name_dic[class_name]
        index = name_list.index(class_name)
        res[index].append([*item['xyxy'], prob])
    for i, item in enumerate(res):
        res[i] = np.array(item)
    return res


def mergeGtAndPred(ann1_dic, ann2_dic, name_list, iou_th=0.5):
    gt_list = getBBoxFromObjList(ann1_dic, name_list, prob=1, )
    pred_list = getBBoxFromObjList(ann2_dic, name_list, prob=0.5)
    res = []
    res_obj_list = []
    for i, gt_np in enumerate(gt_list):
        pred_np = pred_list[i]
        if len(gt_np) > 0 and len(pred_np) > 0:
            det_np = np.vstack((gt_np, pred_np))
            _, kep_indes = nms(det_np, iou_th)
            res.append(det_np[kep_indes])
        elif len(gt_np) > 0:
            res.append(gt_np)
        else:
            res.append(pred_np)
    for i, item in enumerate(res):
        obj_name = name_list[i]
        for j, xyxycs in enumerate(item):
            x0, y0, x1, y1, _ = xyxycs
            res_obj_list.append({'name': obj_name, 'xyxy': (x0, y0, x1, y1)})

    ann1_dic['objs_list'] = res_obj_list
    ann1_dic['objs_num'] = len(res_obj_list)
    return ann1_dic


def mergeXml(gt_dir, pre_dir, save_dir, xml_writer, name_list, with_img=False):
    os.makedirs(save_dir, exist_ok=True)
    print('xml save path:', save_dir)
    img_list = getFileListFromDir(gt_dir)
    count = 0
    for img_pth in tqdm(img_list):
        # img_pth = '/workspace/dataset/detection/nanjing_wenyang_0525/resize_15_all/imgs/04974.jpg'
        img_dir, img_name = os.path.split(img_pth)
        endfix = os.path.splitext(img_name)[-1]
        xml_name = img_name.replace(endfix, '.xml')
        sour_xml_pth = os.path.join(gt_dir, xml_name)
        tar_xml_pth = os.path.join(pre_dir, xml_name)
        if (not os.path.exists(sour_xml_pth)) or (not os.path.exists(tar_xml_pth)):
            print('not find xml:', tar_xml_pth)
            continue
        ann1 = xml_writer.readXml(sour_xml_pth)
        ann2 = xml_writer.readXml(tar_xml_pth)
        if (ann1 is not None) and (ann2 is not None):
            ann_merge = mergeGtAndPred(ann1, ann2, name_list)
        elif ann1 is None:
            ann_merge = ann2
        else:
            ann_merge = ann1
        xml_save_pth = os.path.join(save_dir, xml_name)
        xml_writer.writeXml(ann_merge, xml_save_pth)
        if with_img:
            img_save_pth = os.path.join(save_dir, img_name)
            shutil.copy(img_pth, img_save_pth)
        count += 1
        if count == 10 and DEBUG:
            break


if __name__ == '__main__':
    ## use mode to generate predict xml
    DEBUG = False
    err_name_dic = {}

    conf_th = 0.5
    with_img = False

    img_dir = '/datasets/workspace_shawn/gesture_anker/images/gesture_0624'
    save_dir = '/datasets/workspace_shawn/gesture_anker/images/gesture_0624_label'

    config_file = '/workspace/workspace/detection/mmdetection/configs_shawn/gfl/xz_gfl_mobile_fpn_mstrain_2x_face_hand.py'
    checkpoint_file = '/workspace/workspace/detection/mmdetection/work_dirs/xz_gfl_mobile_fpn_mstrain_2x_face_hand/epoch_24.pth'
    # class_names = ('face', 'lefthand', 'righthand')
    class_names = []

    if len(class_names) == 0:
        cfg = Config.fromfile(config_file)
        dataset = build_dataset(cfg.data.test)
        class_names = dataset.CLASSES
    print("class_names: ",class_names)

    model = model_init(config_file,checkpoint_file)
    xml_writer = Xml()

    os.makedirs(save_dir, exist_ok=True)
    img_list = getFileListFromDir(img_dir)
    count = 0
    for img_pth in tqdm(img_list):
        img = cv2.imread(img_pth)
        img_whc = (img.shape[1], img.shape[0], 3)
        img_dir, img_name = os.path.split(img_pth)
        endfix = os.path.splitext(img_name)[-1]
        xml_save_pth = os.path.join(save_dir, img_name.replace(endfix,'.xml'))
        json_save_pth = os.path.join(save_dir, img_name.replace(endfix,'.json'))
        
        img_save_pth = os.path.join(save_dir, img_name)
        if img is not None:
            result = detect(model,img, split=False)
            # xml_writer.mmdetResultToXml(result, class_names, img_pth, xml_save_pth, img_whc, conf_th, is_xyxy=True)
            xml_writer.mmdetResultToLabelMeJson(result, class_names, img_pth, json_save_pth, img_whc, conf_th, is_xyxy=True)
            if with_img:
                shutil.copy(img_pth,img_save_pth)
        count += 1
        if count == 20 and DEBUG:
            break
