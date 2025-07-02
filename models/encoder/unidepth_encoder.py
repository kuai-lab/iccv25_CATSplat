import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from models.encoder.resnet_encoder import ResnetEncoder
from models.decoder.resnet_decoder import ResnetDecoder, ResnetDepthDecoder
from models.encoder.transformer import TransformerBlock

from models.encoder.layers import BackprojectDepth

from pointnet.source.model import PointNet
import fpsample
import numpy as np


class UniDepthExtended(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.unidepth = torch.hub.load(
            "lpiccinelli-eth/UniDepth", "UniDepth", version=cfg.model.depth.version, 
            backbone=cfg.model.depth.backbone, pretrained=True, trust_repo=True, 
            force_reload=True
        )

        self.parameters_to_train = []
        if cfg.model.backbone.name == "resnet":
            self.encoder = ResnetEncoder(
                num_layers=cfg.model.backbone.num_layers,
                pretrained=cfg.model.backbone.weights_init == "pretrained",
                bn_order=cfg.model.backbone.resnet_bn_order,
            )
            # change encoder to take depth as conditioning
            if cfg.model.backbone.depth_cond:
                self.encoder.encoder.conv1 = nn.Conv2d(
                    4,
                    self.encoder.encoder.conv1.out_channels,
                    kernel_size = self.encoder.encoder.conv1.kernel_size,
                    padding = self.encoder.encoder.conv1.padding,
                    stride = self.encoder.encoder.conv1.stride
                )
            self.parameters_to_train += [{"params": self.encoder.parameters()}]
            models = {}
            if cfg.model.gaussians_per_pixel > 1:
                models["depth"] = ResnetDepthDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train +=[{"params": models["depth"].parameters()}]
            for i in range(cfg.model.gaussians_per_pixel):
                models["gauss_decoder_"+str(i)] = ResnetDecoder(cfg=cfg,num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train += [{"params": models["gauss_decoder_"+str(i)].parameters()}]
                if cfg.model.one_gauss_decoder:
                    break
            self.models = nn.ModuleDict(models)
        
        # Load Transformer
        self.transformer = TransformerBlock(
            d_model=self.encoder.num_ch_enc,
            nhead=8,
            dropout=0.1,
            max_len=160*224
        )
        self.parameters_to_train +=[{"params": self.transformer.parameters()}]
        
        # Load PointNet
        self.pointnet = PointNet()
        self.parameters_to_train +=[{"params": self.pointnet.parameters()}]

        pointnet_ckpt_dir = '../../../pointnet/ckpt/save.pth'
        pointnet_ckpt = torch.load(pointnet_ckpt_dir)
        self.pointnet.load_state_dict(pointnet_ckpt)
        
        # Freeze PointNet parameters
        for param in self.pointnet.parameters():
            param.requires_grad = False

        self.backproject_depth = BackprojectDepth(
            cfg.data_loader.batch_size, 320, 448
        )

    def get_parameter_groups(self):
        # only the resnet encoder and gaussian parameter decoder are optimisable
        return self.parameters_to_train
    
    def forward(self, inputs):
        # prediting the depth for the first layer with pre-trained depth
        if ('unidepth', 0, 0) in inputs.keys() and inputs[('unidepth', 0, 0)] is not None:
            depth_outs = dict()
            depth_outs["depth"] = inputs[('unidepth', 0, 0)]
        else:
            with torch.no_grad():
                intrinsics = inputs[("K_src", 0)] if ("K_src", 0) in inputs.keys() else None
                depth_outs = self.unidepth.infer(inputs["color_aug", 0, 0], intrinsics=intrinsics)
        outputs_gauss = {}

        outputs_gauss[("K_src", 0)] = inputs[("K_src", 0)] if ("K_src", 0) in inputs.keys() else depth_outs["intrinsics"]
        outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(outputs_gauss[("K_src", 0)])

        if self.cfg.model.backbone.depth_cond:
            # division by 20 is to put depth in a similar range to RGB
            input = torch.cat([inputs["color_aug", 0, 0], depth_outs["depth"] / 20.0], dim=1)
        else:
            input = inputs["color_aug", 0, 0]

        encoded_features = self.encoder(input)
        b, c, h, w = encoded_features[-1].shape

        # =========== Depth to point cloud ===========
        depth = depth_outs["depth"] # [B, 1, 320, 448]
        inv_K = outputs_gauss[("inv_K_src", 0)] # [B, 3, 3]

        xyz = self.backproject_depth(depth, inv_K)  # [B, 4, 320*448]
        xyz = xyz[:, :-1].permute(0, 2, 1)  # [B, 320*448, 3]; raw points from Unidepth
        
        all_sampled_xyz = []
        for bsz in range(b):
            xyz_sampled_idx = fpsample.bucket_fps_kdline_sampling(xyz[bsz].cpu().numpy(), 3000, h=5)
            xyz_sampled = xyz[bsz, xyz_sampled_idx.astype(np.int64)].unsqueeze(0)
            all_sampled_xyz.append(xyz_sampled)
        all_sampled_xyz = torch.cat(all_sampled_xyz, dim=0)
        all_sampled_xyz = all_sampled_xyz.permute(0, 2, 1) #[B, 3, num_points]

        # =========== Pointnet ===========
        pointnet_out = self.pointnet(all_sampled_xyz)  # [B, 1024, 320*448]
        pointnet_out = pointnet_out.permute(0, 2, 1)

        llava_feats = inputs[('llava_feat', 0)]
        
        # =========== Transformer ===========
        trans_outputs, ca_outputs = self.transformer(encoded_features, llava_feats.float(), pointnet_out)
        
        num_ch = len(self.encoder.num_ch_enc)
        for ti, trans_out in enumerate(trans_outputs):
            encoded_features[num_ch-ti-1] = trans_out

        # predict multiple gaussian depths
        if self.cfg.model.gaussians_per_pixel > 1:
            depth = self.models["depth"](encoded_features)
            depth[("depth", 0)] = rearrange(depth[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel - 1)
            depth[("depth", 0)] = torch.cumsum(torch.cat((depth_outs["depth"][:,None,...], depth[("depth", 0)]), dim=1), dim=1)
            outputs_gauss[("depth", 0)] = rearrange(depth[("depth", 0)], "b n c ... -> (b n) c ...", n = self.cfg.model.gaussians_per_pixel)
        else:
            outputs_gauss[("depth", 0)] = depth_outs["depth"]
        # predict multiple gaussian parameters
        gauss_outs = dict()
        for i in range(self.cfg.model.gaussians_per_pixel):
            outs = self.models["gauss_decoder_"+str(i)](encoded_features)
            if self.cfg.model.one_gauss_decoder:
                gauss_outs |= outs
                break
            else:
                for key, v in outs.items():
                    gauss_outs[key] = outs[key] if i==0 else torch.cat([gauss_outs[key], outs[key]], dim=1)
        for key, v in gauss_outs.items():
            gauss_outs[key] = rearrange(gauss_outs[key], 'b n ... -> (b n) ...')
        outputs_gauss |= gauss_outs

        return outputs_gauss