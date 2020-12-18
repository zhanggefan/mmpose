import numpy as np
import copy

from mmpose.datasets.registry import PIPELINES
from .bottom_up_transform import BottomUpGenerateTarget, HeatmapGenerator, JointsEncoder


class UnbiasHeatmapGenerator(HeatmapGenerator):
    """Generate heatmaps for bottom-up models.

    Args:
        num_joints (int): Number of keypoints
        output_res (int): Size of feature map
        sigma (int): Sigma of the heatmaps.
    """

    def __init__(self, output_res, num_joints, sigma=-1):
        if not isinstance(output_res, (tuple, list)):
            output_res = (output_res, output_res)
        self.W, self.H = output_res
        self.num_joints = num_joints
        if not isinstance(sigma, (tuple, list)):
            sigma = [sigma] * num_joints
        for i in range(len(sigma)):
            if sigma[i] < 0:
                sigma[i] = (self.W * self.H) ** 0.5 / 64
        self.sigma = sigma

    def __call__(self, joints):
        """Generate heatmaps."""
        hms = np.zeros((self.num_joints, self.H, self.W), dtype=np.float32)
        for p in joints:
            for idx, pt in enumerate(p):
                sigma = self.sigma
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or x >= self.W or y >= self.H:
                        continue

                    ul = int(np.round(x - 4 * sigma - 1)), int(np.round(y - 4 * sigma - 1))
                    br = int(np.round(x + 4 * sigma + 2)), int(np.round(y + 4 * sigma + 2))

                    xmin, xmax = max(0, ul[0]), min(br[0], self.W)
                    ymin, ymax = max(0, ul[1]), min(br[1], self.H)

                    diff_x_local, diff_y_local = np.arange(xmin, xmax) + 0.5 - x, np.arange(ymin, ymax) + 0.5 - y
                    hms_local = np.exp(-(diff_x_local[None, :] ** 2 + diff_y_local[:, None] ** 2) / (2 * sigma ** 2))
                    hms[idx, xmin:xmax, ymin:ymax] = np.maximum(hms[idx, xmin:xmax, ymin:ymax], hms_local)
        return hms


class JointsXYVEncoder(object):
    """Encodes the visible joints into (coordinates, score); The coordinate of
    one joint and its score are of `int` type.

    (idx * output_res**2 + y * output_res + x, 1) or (0, 0).

    Args:
        max_num_people(int): Max number of people in an image
        num_joints(int): Number of keypoints
        output_res(int): Size of feature map
        tag_per_joint(bool):  Option to use one tag map per joint.
    """

    def __init__(self, max_num_people, num_joints, *args, **kwargs):
        self.max_num_people = max_num_people
        self.num_joints = num_joints

    def __call__(self, joints):
        """
        Note:
            number of people in image: N
            number of keypoints: K
            max number of people in an image: M

        Args:
            joints (np.ndarray[NxKx3])

        Returns:
            visible_kpts (np.ndarray[MxKx2]).
        """
        res = np.zeros((self.max_num_people, self.num_joints, 3), dtype=np.float32)
        res[:len(joints), :, :2] = joints[:, :, :2]
        res[:len(joints), :, 2] = np.where(joints[:, :, 2] > 0, 1, 0)
        return res


@PIPELINES.register_module()
class UnbiasBottomUpGenerateTarget(BottomUpGenerateTarget):
    """Generate multi-scale heatmap target for bottom-up.

    Args:
        sigma (int): Sigma of heatmap Gaussian
        max_num_people (int): Maximum number of people in an image
    """

    def __init__(self, sigma, max_num_people, unbias_heatmap_encoding=False, joints_xyv_encoding=True):
        super(UnbiasBottomUpGenerateTarget, self).__init__(sigma=sigma,
                                                           max_num_people=max_num_people)
        self.unbias_heatmap_encoding = unbias_heatmap_encoding
        self.joints_xyv_encoding = joints_xyv_encoding

    def _generate(self, num_joints, heatmap_size):
        """Get heatmap generator and joint encoder."""
        HMG = UnbiasHeatmapGenerator if self.unbias_heatmap_encoding else HeatmapGenerator
        JE = JointsXYVEncoder if self.joints_xyv_encoding else JointsEncoder
        heatmap_generator = [
            HMG(output_size, num_joints, self.sigma)
            for output_size in heatmap_size
        ]
        joints_encoder = [
            JE(self.max_num_people, num_joints, output_size, True)
            for output_size in heatmap_size
        ]
        return heatmap_generator, joints_encoder

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        heatmap_generator, joints_encoder = \
            self._generate(results['ann_info']['num_joints'],
                           results['ann_info']['heatmap_size'])
        target_list = list()
        img, mask_list, joints_list = results['img'], results['mask'], results[
            'joints']

        for scale_id in range(results['ann_info']['num_scales']):
            target_t = heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = joints_encoder[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            if self.joints_xyv_encoding:
                joints_list[scale_id] = joints_t.astype(np.float32)
            else:
                joints_list[scale_id] = joints_t.astype(np.int32)

        results['img'], results['masks'], results[
            'joints'] = img, mask_list, joints_list
        results['targets'] = target_list

        return results
