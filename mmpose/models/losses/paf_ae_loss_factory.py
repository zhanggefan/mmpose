# ------------------------------------------------------------------------------
# Adapted from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from ..registry import LOSSES
from .multi_loss_factory import MultiLossFactory
import torch_scatter


@LOSSES.register_module()
class MaskHeatmapLoss(nn.Module):
    def __init__(self,
                 positive_hm_threshold=0.01,
                 misslabel_mask_expansion=0.3,
                 misslabel_mask_hw_ratio=2):
        super(MaskHeatmapLoss, self).__init__()
        self.positive_hm_threshold = positive_hm_threshold
        self.misslabel_mask_expansion = misslabel_mask_expansion
        self.misslabel_mask_hw_ratio = misslabel_mask_hw_ratio

    def _genHMMask(self, hm_pred, masks, jointsXYV):
        # todo: move following code(25-63) to a common function that transforms jointsXYV into person's masks
        h, w = masks.shape[-2:]
        num_kp = jointsXYV.shape[2]
        device = masks.device

        hm_masks = masks[:, None, :, :].repeat((1, num_kp, 1, 1))

        # following code is really logically hard to grip, but makes full use of vectorization

        jointsXYV = jointsXYV.clone()

        vis = (jointsXYV[..., 2] > 0).any(dim=2)
        bs_id, person_id = torch.nonzero(vis, as_tuple=True)

        jointsXYV = jointsXYV[bs_id, person_id]
        jointsXY = jointsXYV[..., :2]
        invisible = (jointsXYV[..., 2] <= 0)
        invisible_XY = invisible[..., None].expand_as(jointsXY)

        jointsXY[invisible_XY] = float('inf')
        bb_TL = jointsXY.min(dim=1).values
        jointsXY[invisible_XY] = float('-inf')
        bb_BR = jointsXY.max(dim=1).values

        bb_WH = (bb_BR - bb_TL)
        bb_WH[bb_WH < 1] = 1
        bb_CT = 0.5 * (bb_BR + bb_TL)

        bb_WH = torch.max(bb_WH, bb_WH[..., [1, 0]] / self.misslabel_mask_hw_ratio)
        bb_BR = (bb_CT + (0.5 + self.misslabel_mask_expansion) * bb_WH).round()
        bb_TL = (bb_CT - (0.5 + self.misslabel_mask_expansion) * bb_WH).round()

        bb_minx, bb_miny = bb_TL[:, 0][:, None, None], bb_TL[:, 1][:, None, None]
        bb_maxx, bb_maxy = bb_BR[:, 0][:, None, None], bb_BR[:, 1][:, None, None]

        grid_y = torch.arange(h, dtype=torch.float32, device=device)[None, :, None]
        grid_x = torch.arange(w, dtype=torch.float32, device=device)[None, None, :]

        person_mask = (grid_x < bb_minx) | (grid_x > bb_maxx) | (grid_y < bb_miny) | (grid_y > bb_maxy)
        # person_mask shape is (num_person, feat_h, feat_w)

        person_i_mask_hm_j = invisible
        # person_i_mask_paf_j shape is (num_person, num_kp)

        person_paf_mask = (person_mask[:, None, ...] | (~invisible[..., None, None])).float()

        hm_masks = torch_scatter.scatter(src=person_paf_mask,
                                         index=bs_id,
                                         dim=0,
                                         out=hm_masks,
                                         reduce='min')
        hm_masks[hm_pred >= self.positive_hm_threshold] = 1.0
        return hm_masks

    def forward(self, hm_pred, jointsXYV, masks, gt):
        hm_masks = self._genHMMask(hm_pred, masks, jointsXYV)
        assert hm_pred.size() == gt.size()
        loss = ((hm_pred - gt) ** 2) * hm_masks
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


@LOSSES.register_module()
class MaskPAFLoss(nn.Module):
    def __init__(self,
                 linkage,
                 positive_distance_threshold=1,
                 misslabel_mask_expansion=0.3,
                 misslabel_mask_hw_ratio=2):
        super(MaskPAFLoss, self).__init__()
        self.linkage = linkage
        self.positive_distance_threshold = positive_distance_threshold
        self.misslabel_mask_expansion = misslabel_mask_expansion
        self.misslabel_mask_hw_ratio = misslabel_mask_hw_ratio

    def _genPAFMask(self, mask, jointsXYV):
        # todo: move following code(101-138) to a common function that transforms jointsXYV into person's masks
        h, w = mask.shape[-2:]
        device = mask.device

        paf_masks = mask[:, None, :, :].repeat((1, len(self.linkage), 1, 1))

        # following code is really logically hard to grip, but makes full use of vectorization

        jointsXYV = jointsXYV.clone()

        vis = (jointsXYV[..., 2] > 0).any(dim=2)
        bs_id, person_id = torch.nonzero(vis, as_tuple=True)

        jointsXYV = jointsXYV[bs_id, person_id]
        jointsXY = jointsXYV[..., :2]
        invisible = (jointsXYV[..., 2] <= 0)
        invisible_XY = invisible[..., None].expand_as(jointsXY)

        jointsXY[invisible_XY] = float('inf')
        bb_TL = jointsXY.min(dim=1).values
        jointsXY[invisible_XY] = float('-inf')
        bb_BR = jointsXY.max(dim=1).values

        bb_WH = (bb_BR - bb_TL)
        bb_WH[bb_WH < 1] = 1
        bb_CT = 0.5 * (bb_BR + bb_TL)

        bb_WH = torch.max(bb_WH, bb_WH[..., [1, 0]] / self.misslabel_mask_hw_ratio)
        bb_BR = (bb_CT + (0.5 + self.misslabel_mask_expansion) * bb_WH).round()
        bb_TL = (bb_CT - (0.5 + self.misslabel_mask_expansion) * bb_WH).round()

        bb_minx, bb_miny = bb_TL[:, 0][:, None, None], bb_TL[:, 1][:, None, None]
        bb_maxx, bb_maxy = bb_BR[:, 0][:, None, None], bb_BR[:, 1][:, None, None]

        grid_y = torch.arange(h, dtype=torch.float32, device=device)[None, :, None]
        grid_x = torch.arange(w, dtype=torch.float32, device=device)[None, None, :]

        person_mask = (grid_x < bb_minx) | (grid_x > bb_maxx) | (grid_y < bb_miny) | (grid_y > bb_maxy)
        # person_mask shape is (num_person, feat_h, feat_w)

        person_i_mask_paf_j = invisible[:, self.linkage].any(dim=-1)
        # person_i_mask_paf_j shape is (num_person, num_paf)

        person_paf_mask = (person_mask[:, None, ...] | (~person_i_mask_paf_j[..., None, None])).float()

        paf_masks = torch_scatter.scatter(src=person_paf_mask,
                                          index=bs_id,
                                          dim=0,
                                          out=paf_masks,
                                          reduce='min')
        # import cv2
        # import numpy as np
        # for bs, paf in enumerate(paf_masks):
        #     for i, m in enumerate(paf.cpu().numpy()):
        #         cv2.imshow(f'{i}', (m * 255).astype(np.uint8))
        #     cv2.waitKey(0)

        return paf_masks

    def _genPAFTarget(self, paf_pred, jointsXYV, mask):
        jointsXYV = jointsXYV.clone()

        paf_loss_weight = self._genPAFMask(mask, jointsXYV)

        bs, c, h, w = paf_pred.shape
        assert c == 2 * len(self.linkage)

        device = paf_pred.device

        linkage = jointsXYV[:, :, self.linkage]
        # joints_linkage shape is (bs, max_p, num_linkage, 2, 3)
        linkage_start = linkage[..., 0, :2]
        linkage_end = linkage[..., 1, :2]
        linkage_v = (linkage[..., 2] > 0).all(dim=-1) & (linkage_start != linkage_end).any(dim=-1)

        # scattered by part
        bs_id, person_id, part_id = torch.nonzero(linkage_v, as_tuple=True)
        linkage_start = linkage_start[linkage_v]
        # linkage_start shape is (num_parts, 2)
        linkage_end = linkage_end[linkage_v]
        # linkage_end shape is (num_parts, 2)
        linkage_vec = (linkage_end - linkage_start)
        linkage_norm = linkage_vec.norm(p=2, dim=-1)
        linkage_vec /= (linkage_norm[..., None])
        # linkage_vec shape is (num_parts, 2)
        vec_x = linkage_vec[:, 0][:, None, None]
        vec_y = linkage_vec[:, 1][:, None, None]
        st_x = linkage_start[:, 0][:, None, None]
        st_y = linkage_start[:, 1][:, None, None]
        xy_norm = linkage_norm[:, None, None]

        grid_y = torch.arange(h, dtype=torch.float32, device=device)[None, :, None]
        grid_x = torch.arange(w, dtype=torch.float32, device=device)[None, None, :]

        x_diff_to_st = grid_x - st_x
        y_diff_to_st = grid_y - st_y

        temp = x_diff_to_st * vec_x + y_diff_to_st * vec_y
        part_mask = (temp >= -self.positive_distance_threshold) & (temp <= (xy_norm + self.positive_distance_threshold))
        temp = x_diff_to_st * vec_y - y_diff_to_st * vec_x
        part_mask &= (temp.abs() <= self.positive_distance_threshold)
        part_mask = part_mask.float()
        # part_mask shape is (num_parts, w, h)
        part_vecmap_x = vec_x * part_mask
        part_vecmap_y = vec_y * part_mask

        scatter_ind = bs_id * len(self.linkage) + part_id

        paf_target_x = paf_pred.new_zeros(size=(bs * len(self.linkage), h, w))
        paf_target_y = paf_pred.new_zeros(size=(bs * len(self.linkage), h, w))
        paf_count = paf_pred.new_zeros(size=(bs * len(self.linkage), h, w))

        paf_count = torch_scatter.scatter(src=part_mask.float(),
                                          index=scatter_ind,
                                          dim=0,
                                          out=paf_count,
                                          reduce='sum')
        paf_target_x = torch_scatter.scatter(src=part_vecmap_x,
                                             index=scatter_ind,
                                             dim=0,
                                             out=paf_target_x,
                                             reduce='sum')
        paf_target_y = torch_scatter.scatter(src=part_vecmap_y,
                                             index=scatter_ind,
                                             dim=0,
                                             out=paf_target_y,
                                             reduce='sum')

        paf_target_x = paf_target_x.reshape((bs, -1, h, w))
        paf_target_y = paf_target_y.reshape((bs, -1, h, w))
        paf_count = paf_count.reshape((bs, -1, h, w))

        paf_divisor = paf_count.clamp_min(1)
        paf_target_x /= paf_divisor
        paf_target_y /= paf_divisor

        paf_target = torch.stack((paf_target_x, paf_target_y), dim=2).reshape_as(paf_pred)
        paf_loss_weight[paf_count > 0] = 1
        paf_loss_weight = torch.stack((paf_loss_weight, paf_loss_weight), dim=2).reshape_as(paf_target)

        # import matplotlib.pyplot as plt
        # for bs_paf in paf_target:
        #     for part in range(len(bs_paf) // 2):
        #         plt.quiver(bs_paf[2 * part].cpu().numpy(),
        #                    bs_paf[2 * part + 1].cpu().numpy(),
        #                    scale=1,
        #                    scale_units='xy')
        #         plt.show()
        return paf_target, paf_loss_weight

    def forward(self, paf_pred, jointsXYV, mask):
        paf_target, paf_loss_weight = self._genPAFTarget(paf_pred, jointsXYV, mask)
        assert paf_pred.size() == paf_target.size() == paf_loss_weight.size()
        loss = ((paf_pred - paf_target) ** 2) * paf_loss_weight
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


@LOSSES.register_module()
class PAFAEMultiLossFactory(MultiLossFactory):
    """Loss for bottom-up models.

    Args:
        num_joints(int): Number of keypoints.
        num_stages(int): Number of stages.
        ae_loss_type(str): Type of ae loss.
        with_ae_loss(list[bool]): Use ae loss or not in multi-heatmap.
        push_loss_factor(list[float]):
            Parameter of push loss in multi-heatmap.
        pull_loss_factor(list[float]):
            Parameter of pull loss in multi-heatmap.
        with_heatmap_loss(list[bool]):
            Use heatmap loss or not in multi-heatmap.
        heatmaps_loss_factor(list[float]):
            Parameter of heatmap loss in multi-heatmap.
    """

    def __init__(self, num_joints, num_stages,
                 with_paf_loss, paf_linkage, paf_loss_factor,
                 ae_loss_type, with_ae_loss, push_loss_factor, pull_loss_factor,
                 with_heatmaps_loss, heatmaps_loss_factor):

        super(PAFAEMultiLossFactory, self).__init__(
            num_joints=num_joints,
            num_stages=num_stages,
            ae_loss_type=ae_loss_type,
            with_ae_loss=with_ae_loss,
            push_loss_factor=push_loss_factor,
            pull_loss_factor=pull_loss_factor,
            with_heatmaps_loss=with_heatmaps_loss,
            heatmaps_loss_factor=heatmaps_loss_factor
        )

        assert isinstance(with_paf_loss, (list, tuple)), \
            'with_paf_loss should be a list or tuple'
        assert isinstance(paf_loss_factor, (list, tuple)), \
            'paf_loss_factor should be a list or tuple'
        assert isinstance(paf_linkage, (list, tuple)), \
            'paf_config should be a list or tuple'

        self.with_paf_loss = with_paf_loss
        self.paf_loss_factor = paf_loss_factor
        self.paf_linkage = paf_linkage

        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    MaskHeatmapLoss()
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in self.with_heatmaps_loss
                ]
            )

        self.paf_loss = \
            nn.ModuleList(
                [
                    MaskPAFLoss(paf_linkage) if with_paf_loss else None
                    for with_paf_loss, paf_linkage in zip(self.with_paf_loss, self.paf_linkage)
                ]
            )

    @classmethod
    def _map_joints(cls, joints, feat_size, ae_tag_per_joint):
        if joints.shape[-1] == 2:
            return joints
        else:
            h, w = feat_size
            num_joints = joints.shape[-2]
            joints_int = joints.long()
            joints_x = joints_int[..., 0]
            joints_y = joints_int[..., 1]
            # and x >= 0 and y >= 0 and x < self.output_res and y < self.output_res
            joints_valid = (joints_x >= 0) & (joints_y >= 0) & (joints_x < w) & (joints_y < h)
            joints_ae_v = joints_int[..., 2] * joints_valid.long()

            if ae_tag_per_joint:
                joints_idx = torch.arange(num_joints, device=joints.device)[None, None, :]
                joints_ae_idx = w * (h * joints_idx + joints_y) + joints_x
            else:
                joints_ae_idx = w * joints_y + joints_x
            return torch.stack([joints_ae_idx, joints_ae_v], dim=-1)

    # todo: arg 'joints' now uses an wasteful encoding, because most image contains much fewer people than M.
    def forward(self, outputs, heatmaps, masks, joints):
        """Forward function to calculate losses.

        Note:
            batch_size: N
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
            num_keypoints: K
            output_channel: C C=2K if use ae loss else K

        Args:
            outputs(List(torch.Tensor[NxCxHxW])): outputs of stages.
            heatmaps(List(torch.Tensor[NxKxHxW])): target of heatmaps.
            masks(List(torch.Tensor[NxHxW])): masks of heatmaps.
            joints(List(torch.Tensor[NxMxKx2])): joints of ae loss.
        """
        heatmaps_losses = []
        push_losses = []
        pull_losses = []
        paf_losses = []
        for idx in range(len(outputs)):
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                offset_feat = self.num_joints
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred,
                                                        joints[idx],
                                                        masks[idx],
                                                        heatmaps[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.paf_loss[idx]:
                # shape (bs, X, 128, 128) or # shape (bs, X, 256, 256)
                paf_channels = 2 * len(self.paf_linkage[idx])
                paf_pred = outputs[idx][:, offset_feat:(offset_feat + paf_channels)]
                offset_feat += paf_channels
                paf_loss = self.paf_loss[idx](paf_pred,
                                              joints[idx],
                                              masks[idx])
                paf_loss = paf_loss * self.paf_loss_factor[idx]

                paf_losses.append(paf_loss)
            else:
                paf_losses.append(None)

            if self.ae_loss[idx]:
                feat_size = outputs[idx].shape[-2:]
                tags_pred = outputs[idx][:, offset_feat:]
                ae_tag_per_joint = (tags_pred.shape[0] > 1)
                batch_size = tags_pred.size()[0]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                joints_ae = self._map_joints(joints[idx], feat_size, ae_tag_per_joint)
                push_loss, pull_loss = self.ae_loss[idx](tags_pred, joints_ae)
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                push_losses.append(None)
                pull_losses.append(None)

        return heatmaps_losses, push_losses, pull_losses, paf_losses
