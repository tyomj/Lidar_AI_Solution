# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys; sys.path.insert(0, "./CenterPoint")

import os
import pickle
import torch
import onnx
import argparse
from onnxsim import simplify
import numpy as np
from torch import nn

from mmdet3d.models.layers import nms_bev
from mmengine.runner import Runner
from mmengine.config import Config
 

def simplify_model(model_path):
    model = onnx.load(model_path)
    if model is None:
        print("File %s is not find! "%model_path)
    return simplify(model)

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default='tool/checkpoint/epoch_20.pth', action='store',
                        type=str, help='checkpoint')
    parser.add_argument('--config', dest='config',
                        default='CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py', action='store',
                        type=str, help='config')
    parser.add_argument("--save-onnx", type=str, default="rpn_centerhead_sim.onnx", help="output onnx")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    return args

class CenterPointVoxelNet_Post(nn.Module):
    def __init__(self, model):
        super(CenterPointVoxelNet_Post, self).__init__()
        self.model = model
        assert( len(model.pts_bbox_head.task_heads) == 6 )

    def forward(self, x):
        x = self.model.pts_backbone(x)
        x = self.model.pts_neck(x)
        x = self.model.pts_bbox_head.shared_conv(x[0])
        pred = [ task(x) for task in self.model.pts_bbox_head.task_heads ]
        return pred[0]['reg'], pred[0]['height'], pred[0]['dim'], pred[0]['rot'], pred[0]['vel'], pred[0]['heatmap'], \
               pred[1]['reg'], pred[1]['height'], pred[1]['dim'], pred[1]['rot'], pred[1]['vel'], pred[1]['heatmap'], \
               pred[2]['reg'], pred[2]['height'], pred[2]['dim'], pred[2]['rot'], pred[2]['vel'], pred[2]['heatmap'], \
               pred[3]['reg'], pred[3]['height'], pred[3]['dim'], pred[3]['rot'], pred[3]['vel'], pred[3]['heatmap'], \
               pred[4]['reg'], pred[4]['height'], pred[4]['dim'], pred[4]['rot'], pred[4]['vel'], pred[4]['heatmap'], \
               pred[5]['reg'], pred[5]['height'], pred[5]['dim'], pred[5]['rot'], pred[5]['vel'], pred[5]['heatmap']

def predict(reg, hei, dim, rot, vel, hm, test_cfg):
    """decode, nms, then return the detection result.
    """
    # convert N C H W to N H W C
    reg = reg.permute(0, 2, 3, 1).contiguous()
    hei = hei.permute(0, 2, 3, 1).contiguous()
    dim = dim.permute(0, 2, 3, 1).contiguous()
    rot = rot.permute(0, 2, 3, 1).contiguous()
    vel = vel.permute(0, 2, 3, 1).contiguous()
    hm = hm.permute(0, 2, 3, 1).contiguous()

    hm = torch.sigmoid(hm)
    dim = torch.exp(dim)

    rot = torch.atan2(rot[..., 0:1], rot[..., 1:2])

    batch, H, W, num_cls = hm.size()

    reg = reg.reshape(batch, H*W, 2)
    hei = hei.reshape(batch, H*W, 1)

    rot = rot.reshape(batch, H*W, 1)
    dim = dim.reshape(batch, H*W, 3)
    hm = hm.reshape(batch, H*W, num_cls)
    vel = vel.reshape(batch, H*W, 2)

    ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
    ys = ys.view(1, H, W).repeat(batch, 1, 1).to(hm)
    xs = xs.view(1, H, W).repeat(batch, 1, 1).to(hm)

    xs = xs.view(batch, -1, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, -1, 1) + reg[:, :, 1:2]

    xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
    ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

    box_preds = torch.cat([xs, ys, hei, dim, vel, rot], dim=2)

    box_preds = box_preds[0]
    hm_preds = hm[0]

    scores, labels = torch.max(hm_preds, dim=-1)

    score_mask = scores > test_cfg.score_threshold

    post_center_range = test_cfg.post_center_limit_range

    if len(post_center_range) > 0:
        post_center_range = torch.tensor(
            post_center_range,
            dtype=hm.dtype,
            device=hm.device,
        )
    distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
        & (box_preds[..., :3] <= post_center_range[3:]).all(1)

    mask = distance_mask & score_mask

    box_preds = box_preds[mask]
    scores = scores[mask]
    labels = labels[mask]

    boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

    selected = nms_bev(boxes_for_nms.float(), scores.float(),
                        thresh=test_cfg.nms.nms_iou_threshold,
                        pre_maxsize=test_cfg.nms.nms_pre_max_size,
                        post_max_size=test_cfg.nms.nms_post_max_size)

    ret = {}
    ret["box3d_lidar"] = box_preds[selected]
    ret["scores"] = scores[selected]
    ret["label_preds"] = labels[selected]

    return ret

def main(args):
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.default_hooks.checkpoint = None
    cfg.work_dir = './work_dirs'
    cfg.load_from = args.checkpoint
    runner = Runner.from_cfg(cfg)
    model = runner.model
    model = model.eval().cuda()
    data_iter = iter(runner.train_dataloader)
    data_batch = next(data_iter)

    post_model = CenterPointVoxelNet_Post(model)
    post_model = post_model.eval().cuda()
    if args.half:
        model = model.half()
        post_model = post_model.half()

    with torch.no_grad():
        data_batch = model.data_preprocessor(data_batch)
        batch_inputs_dict = data_batch["inputs"]
        batch_data_samples = data_batch["data_samples"]
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        voxel_dict = batch_inputs_dict.get('voxels', None)
        points = batch_inputs_dict.get('points', None)
        voxel_features = model.pts_voxel_encoder(voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors'], None, batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        if args.half:
            x = model.pts_middle_encoder(voxel_features.half(), voxel_dict['coors'].half(), batch_size)
        else:
            x = model.pts_middle_encoder(voxel_features, voxel_dict['coors'], batch_size)

        rpn_input  = torch.zeros(x.shape, dtype=torch.float32, device=torch.device("cuda"))
        if args.half:
            rpn_input  = rpn_input.half()

        torch.onnx.export(post_model, rpn_input, "tmp.onnx",
            export_params=True, opset_version=11, do_constant_folding=True,
            keep_initializers_as_inputs=False, input_names = ['input'],
            output_names = ['reg_0', 'height_0', 'dim_0', 'rot_0', 'vel_0', 'hm_0',
                            'reg_1', 'height_1', 'dim_1', 'rot_1', 'vel_1', 'hm_1',
                            'reg_2', 'height_2', 'dim_2', 'rot_2', 'vel_2', 'hm_2',
                            'reg_3', 'height_3', 'dim_3', 'rot_3', 'vel_3', 'hm_3',
                            'reg_4', 'height_4', 'dim_4', 'rot_4', 'vel_4', 'hm_4',
                            'reg_5', 'height_5', 'dim_5', 'rot_5', 'vel_5', 'hm_5'],
            )

        sim_model, check = simplify_model("tmp.onnx")
        if not check:
            print("[ERROR]:Simplify %s error!"% "tmp.onnx")
        onnx.save(sim_model, args.save_onnx)
        print("[PASS] Export ONNX done.")
if __name__ == "__main__":
    args = arg_parser()
    main(args)