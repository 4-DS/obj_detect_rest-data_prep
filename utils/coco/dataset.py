from .coco import load_coco_file
# from mmdet.datasets import build_dataset


def build_dataset(cfg, default_args=None):
    from mmengine.dataset import ClassBalancedDataset
    from mmdet.registry import DATASETS
    from mmdet.datasets.dataset_wrappers import MultiImageMixDataset
    from mmdet.datasets.dataset_wrappers import ConcatDataset
    
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    else:
        dataset = DATASETS.build(cfg, default_args=default_args)
    return dataset


def get_coco_dataset(coco_file, pipeline, filter_empty_gt=False, classes=None):
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
