import copy
import os
import random
from collections import Sequence
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset


def cv2_imshow(img, title=None):
    plt.figure(figsize=(15, 10))

    if title:
        plt.title(title)

    plt.imshow(img)
    plt.show()


def retrieve_data_cfg(cfg, skip_type, is_train=True):
    sub_cfg = copy.deepcopy(cfg)

    def skip_pipeline_steps(config):
        config["pipeline"] = [
            x for x in config.pipeline if x["type"] not in skip_type]

        deep_indexes = [i for i, x in enumerate(
            config.pipeline) if x.get("transforms")]

        for i in deep_indexes:
            config["pipeline"][i]["transforms"] = [
                x for x in config["pipeline"][i]["transforms"] if x["type"] not in skip_type
            ]

    if is_train:
        train_data_cfg = sub_cfg.data.train
    else:
        train_data_cfg = sub_cfg.data.val

    while "dataset" in train_data_cfg and train_data_cfg["type"] != "MultiImageMixDataset":
        train_data_cfg = train_data_cfg["dataset"]

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)
    print(train_data_cfg)

    return sub_cfg


def test_mask_to_images(masks, img_shape, scale_factor):
    def build_mask(seg):
        out_mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.drawContours(
            out_mask, [(np.array(s).reshape(-1, 2) * scale_factor[:2]
                        ).astype(np.int32) for s in seg], -1, 255, -1
        )
        return out_mask

    return [build_mask(seg) for seg in masks]


def display_once(
    cfg,
    count=1,
    is_train=False,
    bbox_color=(255, 102, 61),
    text_color=(255, 102, 61),
    show=True,
    font_size=14,
    show_interval=0,
    skip_type=["DefaultFormatBundle", "ImageToTensor",
               "Normalize", "Collect", "FilterAnnotations"],
):
    out_cfg = retrieve_data_cfg(cfg, skip_type=skip_type, is_train=is_train)

    if not is_train:
        dataset = build_dataset(out_cfg.data.val)
    else:
        dataset = build_dataset(out_cfg.data.train)

    dataset_len = len(dataset)
    print(f"{dataset_len=}")

    item_idx = [random.randint(0, dataset_len - 1) for _ in range(count)]

    use_mask = "gt_masks" in cfg["train_pipeline"][-1]["keys"]
    gt_masks = None

    l = 0
    for item_id in item_idx:
        item = dataset[item_id]

        if use_mask:
            gt_masks = item.get("gt_masks", None)

            if gt_masks is not None:
                gt_masks = mask2ndarray(gt_masks)

        if item.get("gt_bboxes") is None:
            item["gt_bboxes"] = np.zeros((0, 4))
            item["gt_labels"] = np.zeros((0))

            if item.get("ann_info", [{}])[0].get("bboxes") is not None:
                pad_fixed_size = item["scale_factor"][0]

                scale_factor = item["scale_factor"][0]
                item["gt_bboxes"] = item["ann_info"][0]["bboxes"] * scale_factor
                item["gt_labels"] = item["ann_info"][0]["labels"]
                item["img"] = item["img"][0]
                if use_mask:
                    gt_masks = test_mask_to_images(
                        item["ann_info"][0]["masks"], item["img"].shape, scale_factor)

        if gt_masks is not None:
            gt_masks = np.array(gt_masks)

        out_image = imshow_det_bboxes(
            item["img"],
            item["gt_bboxes"],
            item["gt_labels"],
            gt_masks,
            class_names=dataset.CLASSES,
            show=False,
            font_size=font_size,
            wait_time=show_interval,
            bbox_color=bbox_color,
            text_color=text_color,
        )

        if show:
            cv2_imshow(out_image[:, :, ::-1], title=item["filename"])

        l += 1
        if l >= count:
            break
