#-*-coding:utf-8-*-
import pickle, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import itertools
from terminaltables import AsciiTable

def read_pickle(pkl):
    with open(pkl,'rb') as f:
        data = pickle.load(f)
    return data

def read_json(json_pth):
    with open(json_pth,'r') as f:
        data = json.load(f)
    return data

iou_th = 0.5
# det_json = '/workspace/code/mmdet/mmdetection_xz/work_dir_mmdet/1118_cascade_rd_fi_101dcn/20201120_005009_ep18.json'
det_json = '/workspace/code/mmdet/mmdetection_xz/work_dir_mmdet/1118_cascade_rd_fi_101dcn/20201209_022925_ep22_fix.json'
gt_json = '/workspace/dataset/detection/nanjing_wenyang_0525/final_15_class/annotations/imgs_sp900_rd_test_fix_clean.json'
# det_json = '/workspace/code/mmdet/mmdetection_xz/work_dir_mmdet/1116_cascade_rd_original_4k/20201117_051829.json'
# gt_json = '/workspace/dataset/detection/nanjing_wenyang_0525/original_name_15_class/annotations/imgs_sp900_rd_test.json'

det_data = read_json(det_json)
gt_data = read_json(gt_json)
print('det:',det_json)
print('gt:',gt_json)
cocoGt = COCO(gt_json)
cocoDt = cocoGt.loadRes(det_json)
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.params.iouThrs = np.linspace(iou_th, iou_th+0.45, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
cocoEval.params.maxDets = list((100, 300, 1000))
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

classwise = True
precisions = cocoEval.eval['precision'] # TP/(TP+FP) right/detection
recalls = cocoEval.eval['recall'] # iou*class_num*Areas*Max_det TP/(TP+FN) right/gt
pre = precisions[0, :, :, 0, -1]
print('*'*10)
print(pre.shape)
print(np.mean(pre[:,:14]))
print(np.mean(pre[:,:15]))
print(np.mean(pre[:,np.array([0,1,2,3,4,5,6,7,8,9,10,12,13])]))
print(np.mean(pre[:,np.array([0,1,2,3,4,5,6,7,8,9,11,12,13])]))

print('\nIOU:{} MAP:{:.3f} Recall:{:.3f}'.format(cocoEval.params.iouThrs[0],np.mean(precisions[0, :, :, 0, -1]),np.mean(recalls[0, :, 0, -1])))
CLASSES = ('feng', 'hehua', 'hudie', 'juhua', 'laohu', 'long', 'meihua', 'mudan',
               'nvren', 'qilin', 'ruyi', 'wan', 'yinghai', 'yun', 'yuwen')
class_num = len(CLASSES)

if classwise:  # Compute per-category AP
    # Compute per-category AP
    # from https://github.com/facebookresearch/detectron2/
    # precision: (iou, recall, cls, area range, max dets)

    results_per_category = []
    results_per_category_iou50 = []
    res_item = []
    for idx, catId in enumerate(range(class_num)):
        name = CLASSES[idx]
        precision = precisions[:, :, idx, 0, -1]
        precision_50 = precisions[0, :, idx, 0, -1]
        precision = precision[precision > -1]

        recall = recalls[ :, idx, 0, -1]
        recall_50 = recalls[0, idx, 0, -1]
        recall = recall[recall > -1]

        if precision.size:
            ap = np.mean(precision)
            ap_50 = np.mean(precision_50)
            rec = np.mean(recall)
            rec_50 = np.mean(recall_50)
        else:
            ap = float('nan')
            ap_50 = float('nan')
            rec = float('nan')
            rec_50 = float('nan')
        res_item = [f'{name}', f'{float(ap):0.3f}',f'{float(rec):0.3f}']
        results_per_category.append(res_item)
        res_item_50 = [f'{name}', f'{float(ap_50):0.3f}', f'{float(rec_50):0.3f}']
        results_per_category_iou50.append(res_item_50)

    item_num = len(res_item)
    num_columns = min(6, len(results_per_category) * item_num)
    results_flatten = list(
        itertools.chain(*results_per_category))
    headers = ['category', 'AP', 'Recall'] * (num_columns // item_num)
    results_2d = itertools.zip_longest(*[
        results_flatten[i::num_columns]
        for i in range(num_columns)
    ])
    table_data = [headers]
    table_data += [result for result in results_2d]
    table = AsciiTable(table_data)
    print('\n' + table.table)

    num_columns_50 = min(6, len(results_per_category_iou50) * item_num)
    results_flatten_50 = list(
        itertools.chain(*results_per_category_iou50))
    iou_ = cocoEval.params.iouThrs[0]
    headers_50 = ['category', 'AP{}'.format(iou_),'Recall{}'.format(iou_)] * (num_columns_50 // item_num)
    results_2d_50 = itertools.zip_longest(*[
        results_flatten_50[i::num_columns_50]
        for i in range(num_columns_50)
    ])

    table_data_50 = [headers_50]
    table_data_50 += [result for result in results_2d_50]
    table_50 = AsciiTable(table_data_50)
    print('\n' + table_50.table)
