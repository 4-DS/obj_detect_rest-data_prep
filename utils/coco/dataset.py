from .coco import load
from mmdet.datasets import build_dataset


def get_dataset(coco_file, pipeline, filter_empty_gt=False, classes=None):
    if classes is None:
        _coco_data = load(coco_file)
        classes = [_cat['name'] for _cat in _coco_data['categories']]

    dataset_pipeline = dict(
        type='CocoDataset',
        filter_empty_gt=filter_empty_gt,  # for empty (wht obj)
        img_prefix='',
        ann_file=coco_file,
        pipeline=pipeline,
        classes=classes,
    )

    return build_dataset(dataset_pipeline)
