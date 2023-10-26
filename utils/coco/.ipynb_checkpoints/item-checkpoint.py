import os
import os.path as osp
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mmdet.core.visualization import imshow_det_bboxes
from mmdet.core.utils import mask2ndarray
import mmcv

from .encoder import dump
from .slicer import cut_by_max_size

import logging
logger = logging.getLogger(__name__)


USE_MEAN_MASK = False


def calc_means_and_std(img, bgr=[0, 1, 2]):
    """Это все расчет общего среднего значение пикселя по слоям и дисперсии изображения"""
    np_img = img.copy()

    params = {}
    if USE_MEAN_MASK:
        params['mask'] = np_img[:, :, 0].astype(bool).astype(np.uint8)*255

    means, stds = cv2.meanStdDev(np_img, **params)

    return means[bgr].ravel(), stds[bgr].ravel()


def masks_to_cnt(gt_masks, gt_bboxes, gt_labels, image_id=0):
    result = []
    item_id = 0
    for i, mask in enumerate(gt_masks):
        category_id = gt_labels[i]
        x1, y1, x2, y2 = gt_bboxes[i]

        print(mask)

        if not mask[y1:y2, x1:x2].any():
            continue

        coords, _ = cv2.findContours(mask[y1:y2, x1:x2].astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        coords = list(coords)

        segmentation = []
        if len(coords) > 0:
            for cnt in coords:
                if cnt.shape[0] >= 4:
                    cnt[:, :, [0]] += x1
                    cnt[:, :, [1]] += y1
                    segmentation.append(cnt.ravel())

        if len(segmentation) > 0:
            cnt = np.hstack(segmentation).reshape(-1, 2)
            bx, by, bw, bh = list(cv2.boundingRect(cnt))
            result.append({
                "image_id": image_id,
                "id": item_id,
                "segmentation": segmentation,
                "area": bw * bh,
                "bbox": [bx, by, bw, bh],
                'iscrowd': 0,
                "category_id": category_id,
            })

            item_id += 1

    return result


def poly_to_cnt(gt_masks, gt_bboxes, gt_labels, image_id=0):
    result = []
    item_id = 0
    for i, segmentation in enumerate(gt_masks):
        category_id = gt_labels[i]
        x1, y1, x2, y2 = gt_bboxes[i]

        if len(segmentation) > 0:
            cnt = np.hstack(segmentation).astype(np.int32).reshape(-1, 2)
            bx, by, bw, bh = list(cv2.boundingRect(cnt))
            result.append({
                "image_id": image_id,
                "id": item_id,
                "segmentation": segmentation,
                "area": bw * bh,
                "bbox": [bx, by, bw, bh],
                'iscrowd': 0,
                "category_id": category_id,
            })

            item_id += 1

    return result


png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]


def item_to_coco(dataset, idx, output_dir, max_size, cat_ids, clases, overlap=0.5, save_empty=False, min_gt_bbox_wh=0, save_all_image=False):
    try:
        item_ts = time.time()

        item = dataset[idx]
        os.makedirs(output_dir, exist_ok=True)
        # gt_masks  = mask2ndarray(item["gt_masks"])
        gt_masks = item["gt_masks"]

        gt_boxes = item["gt_bboxes"].astype(np.int32)
        gt_labels = item["gt_labels"]
        if cat_ids:
            gt_labels = [cat_ids[category_id] for category_id in gt_labels]

        filename = osp.basename(item['img_metas'].data['filename'])

        if save_all_image:
            xyxy = np.int0([
                [0, 0, item['img'].shape[1], item['img'].shape[0]]
            ])
        else:
            xyxy = cut_by_max_size(
                item['img_metas'].data['img_shape'], max_size, overlap)

        coco_data = {
            'info': {'year': 2022,
                     'version': '1.0',
                     'description': 'build for arnest dataset',
                     'contributor': 'MiXaiLL76',
                     'url': 'https://t.me/mixaill76',
                     'date_created': '2022-05-13 16:21:15'
                     },
            'images': [],
            'annotations': [],
            'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
            'categories': [{'id': cat_ids[i], 'name': clases[i]} for i in range(len(clases))],
        }

        for x1, y1, x2, y2 in xyxy:
            _filename = osp.join(output_dir, f"{x1}_{y1}_{x2}_{y2}_{filename}")
            _filename = os.path.splitext(_filename)[0] + '.png'

            img = item['img'][y1:y2, x1:x2, :]
            gt_sub_mask = gt_masks.crop(np.array([x1, y1, x2, y2]))

            mean, std = calc_means_and_std(img)

            image_id = 0
            coco_data['images'] = [{'id': image_id,
                                    'width': x2-x1,
                                    'height': y2-y1,
                                    'mean': mean,
                                    'std': std,
                                    'file_name': _filename,
                                    'license': 1,
                                    'date_captured': ''}]

            coco_data['annotations'] = poly_to_cnt(
                gt_sub_mask, gt_boxes - [x1, y1, x1, y1], gt_labels, image_id)
            if min_gt_bbox_wh > 0:
                coco_data['annotations'] = [ann for ann in coco_data['annotations'] if min(
                    ann['bbox'][2], ann['bbox'][3]) >= min_gt_bbox_wh]
                coco_data['annotations'] = [
                    dict(ann, **{'id': i}) for i, ann in enumerate(coco_data['annotations'])]

            if len(coco_data['annotations']) == 0:
                if not save_empty:
                    continue

            cv2.imwrite(_filename, img, png_params)
            coco_file_name = osp.splitext(_filename)[0]+'.json'
            dump(coco_file_name, coco_data)

        item_te = time.time()
        item_total = item_te - item_ts
        logger.debug(f'[{filename}] {item_total=:.3f}')

    except Exception as e:
        logger.exception('stop_on_error', exc_info=True)
        return Exception('stop_on_error')


def show_item(dataset, item_id, bbox_color=(255, 102, 61), text_color=(255, 102, 61), class_names=None, return_image=False, title=None, show_mask=False):
    item = dataset[item_id]
    # print(item)
    
    if show_mask:
        gt_masks = item.get('gt_masks')
        if type(gt_masks) is not np.ndarray:
            gt_masks = mask2ndarray(gt_masks)
    else:
        gt_masks=None

    if class_names is None:
        class_names = dataset.CLASSES
    
    gt_bboxes = item["gt_bboxes"]
    gt_labels = item["gt_labels"]
    if len(gt_bboxes)==0:
        gt_bboxes = None
        
    if (gt_bboxes is None) and (gt_masks is None):
        out_image = mmcv.imread(item["img"]).astype(np.uint8)
        out_image = mmcv.bgr2rgb(out_image)
        width, height = out_image.shape[1], out_image.shape[0]
        out_image = np.ascontiguousarray(out_image)    
    else:
        out_image = imshow_det_bboxes(
            item["img"],
            gt_bboxes,
            gt_labels,
            gt_masks,
            class_names=class_names,
            show=False,
            font_size=14,
            wait_time=0,
            bbox_color=bbox_color,
            text_color=text_color,
        )

    if return_image:
        return out_image

    plt.figure(figsize=(10, 10))

    if title is not None:
        plt.title(title)

    plt.imshow(out_image)
    plt.show()

    
    