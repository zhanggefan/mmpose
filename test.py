from mmpose.datasets import build_dataset, build_dataloader
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
import torch
from mmpose.models import build_posenet
import cv2
import numpy as np


def vis_car_pose_result(model,
                        img,
                        result,
                        kpt_score_thr=0.001,
                        show=False,
                        out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0]])

    radius = 4

    # show the results
    skeleton = [[1, 2], [2, 3], [3, 4], [4, 1]]

    pose_limb_color = palette[[
        0, 1, 2, 3
    ]]
    pose_kpt_color = palette[[
        3, 2, 1, 0
    ]]

    img = model.show_result(
        img,
        result,
        skeleton,
        radius=radius,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        show=show,
        out_file=out_file)

    return img


cfg = Config.fromfile('configs/top_down/hrnet/cowacar/cowacar.py')

dataset = build_dataset(cfg.data.test, dict(test_mode=True))
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)

# for d in datasets:
#     img = (255 * d['img'].permute(1, 2, 0).numpy())[:, :, ::-1]
#     hm = d['target'].max(axis=0, keepdims=True).transpose(1, 2, 0)
#     hm = cv2.resize(hm, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
#     img *= 0.8
#     img[..., -1] += 255 * 0.2 * hm
#     cv2.imshow('0', img.astype(np.uint8))
#     cv2.waitKey(0)

model = build_posenet(cfg.model)
_ = load_checkpoint(model, 'work_dirs/cowacar/epoch_6.pth', map_location='cpu')
model = MMDataParallel(model, device_ids=[0])
model.eval()
results = []
for data in data_loader:
    with torch.no_grad():
        result = model(return_loss=False, **data)

    results.append(result)
    pose_results = []
    all_preds, bbox, path, hm = result

    pred = all_preds[0]
    xc, yc, wb, hb = bbox[0, :4]
    wb *= 200
    hb *= 200
    print('wb,hb', xc, yc, wb, hb)
    img = data['img'][0].permute(1, 2, 0).numpy()
    img -= img.min()
    img /= img.max()
    img = 255 * img[:, :, ::-1]
    # hm = hm[0].max(axis=0, keepdims=True).transpose(1, 2, 0)
    # hm = hm.clip(0.001, 1)
    # hm /= hm.max()
    # hm = cv2.resize(hm, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    img = np.ascontiguousarray(img.astype(np.uint8))
    palette = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 255]]
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    for i, (x, y, conf) in enumerate(pred):
        if conf > 0.5:
            x = (x - xc) / wb * 128 + 64
            y = (y - yc) / hb * 96 + 48
            img = cv2.circle(img, (round(4 * x), round(4 * y)), 3, palette[i], -1)
    cv2.imshow('0', img)
    cv2.waitKey(0)

    # #
    # hm = hm[0].transpose(1, 2, 0)
    # hm = cv2.resize(hm, None, fx=16, fy=16, interpolation=cv2.INTER_LINEAR)
    # for i in range(4):
    #     img_hm = (0.5 * img + 0.5 * 255 * hm[..., [i]]).clip(0, 255).astype(np.uint8)
    #     cv2.imshow('0', img_hm)
    #     cv2.waitKey(0)

print(dataset.evaluate(results, './work_dirs/cowacar', metric='mAP'))
