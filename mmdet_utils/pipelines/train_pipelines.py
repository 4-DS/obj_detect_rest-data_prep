import numpy as np
import mmcv
import copy
import logging
import cv2
import os.path as osp
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.loading import FilterAnnotations, LoadImageFromFile
from mmdet.datasets.pipelines import Rotate, Resize

logger = logging.getLogger(__name__)


# TODO:
# https://github.com/open-mmlab/mmdetection/issues/8131

@PIPELINES.register_module()
class FilterAnnotationsBugFix(FilterAnnotations):
    def __call__(self, results):
        if self.by_box:
            assert 'gt_bboxes' in results
            gt_bboxes = results['gt_bboxes']
            instance_num = gt_bboxes.shape[0]
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            instance_num = len(gt_masks)

        if instance_num == 0:
            return results

        tests = []
        if self.by_box:
            w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            tests.append((w > self.min_gt_bbox_wh[0])
                         & (h > self.min_gt_bbox_wh[1]))
        if self.by_mask:
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        # BUG FIX
        keep = keep.nonzero()[0]

        keys = ('gt_bboxes', 'gt_labels', 'gt_masks')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]

        if keep.size == 0:
            if self.keep_empty:
                return None
        return results


@PIPELINES.register_module()
class LoadImageArnest(LoadImageFromFile):
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            image_path = osp.join(results['img_prefix'],
                                  results['img_info']['filename'])
        else:
            image_path = results['img_info']['filename']

        gray = cv2.imread(image_path, 0)

        depth_frame_8 = cv2.imread(
            image_path.replace('grayscale.png', 'depth.png'), 0)
        gray_edges = cv2.Canny(gray, 0, 150, 3)
        img = cv2.merge([gray, depth_frame_8, gray_edges])

        bits_drop_count = 2
        # gray_edges
        img[:, :, 2] = cv2.Canny(img[:, :, 0], 0, 150, 3)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = image_path
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class RandomRotation(Rotate):
    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Rotated results.
        """

        if np.random.rand() > self.prob:
            return results

        base_results = copy.deepcopy(results)

        try:
            h, w = results['img'].shape[:2]

            center = self.center
            if center is None:
                center = ((w - 1) * 0.5, (h - 1) * 0.5)

            angle = np.random.randint(0, self.angle)

            self._rotate_img(results, angle, center, self.scale)
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
            self._rotate_bboxes(results, rotate_matrix)
            self._rotate_masks(results, angle, center, self.scale, fill_val=0)
            self._rotate_seg(
                results, angle, center, self.scale, fill_val=self.seg_ignore_label)
            self._filter_invalid(results)
        except:
            return base_results

        return results


@PIPELINES.register_module()
class FixedCrop:
    def __init__(self, x, y, crop_size,
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True):
        self.x = x
        self.y = y
        self.crop_size = crop_size

        if not isinstance(self.x, list):
            self.x = [self.x]

        if not isinstance(self.y, list):
            self.y = [self.y]

        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox

        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _random_offset(self):
        y_id = np.random.randint(len(self.y))
        x_id = np.random.randint(len(self.x))
        return self.y[y_id], self.x[x_id]

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]

            offset_h, offset_w = self._random_offset()

            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img

        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not self.allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str
