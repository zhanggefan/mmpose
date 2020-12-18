# ------------------------------------------------------------------------------
# Adapted from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from ..registry import LOSSES
from .multi_loss_factory import MultiLossFactory, _make_input


@LOSSES.register_module()
class PAFLoss(nn.Module):
    """
    """

    def __init__(self, linkage=[]):
        super(PAFLoss, self).__init__()
        self.linkage = linkage

    def singlePAFLoss(self, pred_paf, jointsXYV):
        """Associative embedding loss for one image.

        Note:
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
            num_keypoints: K
            num_linkage: L
        Args:
            pred_tag(torch.Tensor[HxWx(Lx2)]): tag of output for one image.
            joints(torch.Tensor[MxKx2]): joints information for one image.
        """
        feat_size = pred_paf.shape[:2]
        device = pred_paf.device
        self._genXYGrid(feat_size, device)

        for joints_per_person in jointsXYV:
            self._genPAFMask(feat_size, jointsXYV)
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp)) ** 2)

        num_tags = len(tags)
        if num_tags == 0:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device),
                    _make_input(
                        torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(),
                device=pred_tag.device), pull / (num_tags))

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return paf_loss

    def _genPAFMask(self, feat_size, jointsXYV):
        pass

    def _genPAFTarget(self, feat_size, jointsXYV):
        pass

    def _genXYGrid(self, feat_size, device):
        if hasattr(self, 'grid_x') and hasattr(self, 'grid_y') and self.grid_x.shape == self.grid_y.shape == feat_size:
            return
        h, w = feat_size
        self.grid_y, self.grid_x = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                                                  torch.arange(w, dtype=torch.float32, device=device))

    def forward(self, paf_pred, joints, mask):
        return 0.
        pafs = []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            paf = self.singlePAFLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes)


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

        self.paf_loss = \
            nn.ModuleList(
                [
                    PAFLoss(self.paf_linkage) if with_paf_loss else None
                    for with_paf_loss in self.with_paf_loss
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
                                                        heatmaps[idx],
                                                        masks[idx])
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                heatmaps_losses.append(heatmaps_loss)
            else:
                heatmaps_losses.append(None)

            if self.paf_loss[idx]:
                # shape (bs, X, 128, 128) or # shape (bs, X, 256, 256)
                paf_channels = len(self.paf_linkage[idx])
                paf_pred = outputs[idx][:, offset_feat:(offset_feat + paf_channels)]
                # offset_feat += paf_channels
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
