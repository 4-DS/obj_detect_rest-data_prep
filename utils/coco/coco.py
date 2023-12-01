import pycocotools._mask as _mask_coco_util
import os
import os.path as osp
import json
import copy
import cv2
import numpy as np
import datetime
from tqdm import tqdm
from .utils import get_files
from .encoder import load_coco_file
from .item import show_coco_dataset_item

import logging
logger = logging.getLogger(__name__)


def segm_to_mask(segm, image):
    # try:
    assert type(segm) is list, 'segm not list'
    if type(segm[0]) is not list:
        segm = [segm]

    rles = _mask_coco_util.frPyObjects(segm, image['height'], image['width'])
    rle = _mask_coco_util.merge(rles)

    return _mask_coco_util.decode([rle])[:, :, 0] * 255


def preview_coco_file(coco_file, img_folder=None, count=1, random=True, return_image=False, img_load_callback=None, max_objects=None):
    if type(coco_file) is str:
        coco_data = load(coco_file)
    elif type(coco_file) is dict:
        coco_data = copy.deepcopy(coco_file)
        if img_folder is None:
            raise Exception('not loaded')
    else:
        raise Exception('not loaded')

    images = coco_data.get('images', [])

    if len(images) > count:
        images = np.random.choice(images, count, replace=False)

    _out = []

    for image in images:
        image_title = image.get('file_name')
        if image.get('numpy') is not None:
            vis_img = image.get('numpy')
        else:
            if img_folder is None:
                img_path = osp.join(osp.dirname(coco_file), image['file_name'])
            else:
                img_path = osp.join(img_folder, image['file_name'])

            vis_img = cv2.imread(img_path)

        if img_load_callback is not None:
            vis_img = img_load_callback(vis_img)

        item = {
            "img": vis_img,
            "gt_masks": [],
            "gt_bboxes": [],
            "gt_labels": [],
        }

        _categories = {_cat['id']: _i for _i,
                       _cat in enumerate(coco_data['categories'])}
        
        count = 0
        for ann_idx, ann in enumerate(coco_data.get('annotations', [])):            
            if max_objects is not None:
                if count > max_objects:
                    break            
            if ann.get('image_id') == image['id']:
                count += 1
                try:
                    item['gt_masks'].append(
                        segm_to_mask(ann["segmentation"], image))
                except:
                    pass

                x1, y1, w, h = ann["bbox"]
                item['gt_bboxes'].append([x1, y1, x1+w, y1+h])
                item['gt_labels'].append(_categories[ann["category_id"]])

        item['gt_masks'] = np.array(item['gt_masks'])
        item['gt_bboxes'] = np.array(item['gt_bboxes'])
        item['gt_labels'] = np.array(item['gt_labels'])

        try:
            _out.append(
                show_coco_dataset_item([item], 0, class_names=[
                          _cat['name'] for _cat in coco_data['categories']], return_image=return_image, title=image_title)
            )
        except Exception as e:
            print(image.get('file_name') + " BAD ITEM NOT SHOWED")
            raise e

    return _out


def join_coco_files(files_dir=None, files=None, output_file=None, min_gt_bbox_wh=0, keep_ratio=(True, 'max'), max_size=None):
    timestamp = datetime.datetime.now()

    all_coco = {
        "info": {
            "year": timestamp.year,
            "version": "1.0",
            "description": "build for arnest dataset",
            "contributor": "MiXaiLL76",
            "url": "https://t.me/mixaill76",
            "date_created": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "images": [],
        "annotations": [],
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [],
    }

    assert files is not None or files_dir is not None, ''

    if files_dir:
        files = []
        if type(files_dir) is list:
            for file_dir in files_dir:
                files += get_files(file_dir, '*.json')
        elif type(files_dir) is str:
            files = get_files(files_dir, '*.json')
        else:
            raise Exception()

    loaded_categories = []
    loaded_images = 0
    loaded_anns = 0
    for file in tqdm(files):
        coco_data = load(file)

        _categories = {_cat['id']: _cat['name']
                       for _cat in coco_data['categories']}
        loaded_categories += list(_categories.values())

        anns = coco_data.get('annotations', [])
        images = coco_data.get('images', [])

        for image in images:
            ratio = (1, 1)
            if max_size is not None:
                if keep_ratio[0]:
                    if keep_ratio[1] == 'min':
                        _ratio = min(image['width'],
                                     image['height']) / max_size
                    else:
                        _ratio = max(image['width'],
                                     image['height']) / max_size

                    ratio = (_ratio, _ratio)
                else:
                    ratio = image['width'] / \
                        max_size, image['height'] / max_size

            # append image
            all_coco['images'].append(dict(image, **{
                'id': loaded_images,
                'file_name': osp.join(osp.dirname(file), image['file_name'])
            }))

            for ann in coco_data.get('annotations', []):
                if ann.get('image_id') == image['id']:
                    if min(ann['bbox'][2] / ratio[0], ann['bbox'][3] / ratio[1]) >= min_gt_bbox_wh:
                        all_coco['annotations'].append(
                            dict(ann, **{
                                'id': loaded_anns,
                                'image_id': loaded_images,
                                'category_id': _categories[ann['category_id']],
                            })
                        )
                        loaded_anns += 1

            loaded_images += 1
    loaded_categories = [{'id': (_i + 1), 'name': _cat}
                         for _i, _cat in enumerate(list(set(loaded_categories)))]
    all_coco['categories'] = loaded_categories
    _loaded_categories = {_cat['name']: _cat['id']
                          for _cat in loaded_categories}

    obj_max_size = 0
    for i, ann in enumerate(all_coco['annotations']):
        all_coco['annotations'][i]['category_id'] = _loaded_categories[ann['category_id']]
        if max(ann['bbox'][2:]) > obj_max_size:
            obj_max_size = max(ann['bbox'][2:])

    logger.info(f"{loaded_images=}")
    logger.info(f"{loaded_anns=}")
    logger.info(f"{loaded_categories=}")
    logger.info(f"{obj_max_size=}")

    if output_file:
        with open(output_file, 'w') as io:
            io.write(json.dumps(all_coco))

        return obj_max_size
    else:
        return all_coco, obj_max_size
