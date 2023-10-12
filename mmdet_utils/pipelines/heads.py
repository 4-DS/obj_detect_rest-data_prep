import torch
import torch.nn.functional as F
from mmdet.models.dense_heads import YOLACTProtonet
from mmcv.runner import BaseModule, ModuleList, force_fp32
from mmdet.models.losses.dice_loss import dice_loss
from mmdet.models.builder import HEADS

if "YOLACTProtonet_Dice" in HEADS.module_dict:
    del HEADS.module_dict['YOLACTProtonet_Dice']


@HEADS.register_module()
class YOLACTProtonet_Dice(YOLACTProtonet):
    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, gt_masks, gt_bboxes, img_meta, sampling_results):
        """Compute loss of the head.
        Args:
            mask_pred (list[Tensor]): Predicted prototypes with shape
                (num_classes, H, W).
            gt_masks (list[Tensor]): Ground truth masks for each image with
                the same shape of the input image.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_meta (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            sampling_results (List[:obj:``SamplingResult``]): Sampler results
                for each image.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_mask = []
        num_imgs = len(mask_pred)
        total_pos = 0
        for idx in range(num_imgs):
            cur_mask_pred = mask_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_bboxes = gt_bboxes[idx]
            cur_img_meta = img_meta[idx]
            cur_sampling_results = sampling_results[idx]

            pos_assigned_gt_inds = cur_sampling_results.pos_assigned_gt_inds
            num_pos = pos_assigned_gt_inds.size(0)
            # Since we're producing (near) full image masks,
            # it'd take too much vram to backprop on every single mask.
            # Thus we select only a subset.
            if num_pos > self.max_masks_to_train:
                perm = torch.randperm(num_pos)
                select = perm[:self.max_masks_to_train]
                cur_mask_pred = cur_mask_pred[select]
                pos_assigned_gt_inds = pos_assigned_gt_inds[select]
                num_pos = self.max_masks_to_train
            total_pos += num_pos

            gt_bboxes_for_reweight = cur_gt_bboxes[pos_assigned_gt_inds]

            mask_targets = self.get_targets(cur_mask_pred, cur_gt_masks,
                                            pos_assigned_gt_inds)
            if num_pos == 0:
                loss = cur_mask_pred.sum() * 0.
            elif mask_targets is None:
                loss = dice_loss(cur_mask_pred, torch.zeros_like(
                    cur_mask_pred), torch.zeros_like(cur_mask_pred))
            else:
                cur_mask_pred = torch.clamp(cur_mask_pred, 0, 1)

                # DEFAULT reduction='none'
                loss = dice_loss(cur_mask_pred, mask_targets,
                                 reduction='mean') * self.loss_mask_weight

                h, w = cur_img_meta['img_shape'][:2]
                gt_bboxes_width = (gt_bboxes_for_reweight[:, 2] -
                                   gt_bboxes_for_reweight[:, 0]) / w
                gt_bboxes_height = (gt_bboxes_for_reweight[:, 3] -
                                    gt_bboxes_for_reweight[:, 1]) / h

                loss = loss / gt_bboxes_width / gt_bboxes_height
                loss = torch.sum(loss)
            loss_mask.append(loss)

        if total_pos == 0:
            total_pos += 1  # avoid nan
        loss_mask = [x / total_pos for x in loss_mask]

        return dict(loss_mask=loss_mask)
