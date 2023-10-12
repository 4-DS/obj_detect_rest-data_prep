import numpy as np
import mmcv
import copy
import logging
import cv2
import os.path as osp
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Resize

logger = logging.getLogger(__name__)


@PIPELINES.register_module()
class FakeFlipInfo:
    def __call__(self, results):
        if results.get('flip') is None:
            results['flip'] = False
            results['flip_direction'] = 'horizontal'
        return results


@PIPELINES.register_module()
class DataAsList:
    def __call__(self, results):
        aug_data_dict = {key: [val] for key, val in results.items()}
        return aug_data_dict


@PIPELINES.register_module()
class ResizeSmallestEdge(Resize):
    def __init__(self, short_edge_length: int):
        self.short_edge_length = short_edge_length
        super().__init__(img_scale=(self.short_edge_length,
                                    self.short_edge_length), keep_ratio=True)

    def resizeShortestEdge(self, results):
        h, w = results['img'].shape[:2]

        scale = self.short_edge_length * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.short_edge_length, scale * w
        else:
            newh, neww = scale * h, self.short_edge_length

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        results['scale'] = (newh, neww)

    def __call__(self, results):
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self.resizeShortestEdge(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self.resizeShortestEdge(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(short_edge_length={self.short_edge_length})'
        return repr_str
