from torch import nn
import torch
import torch.nn.functional as F

from .ppr import PPR
from .mbfn import MobileFacenet
import numpy as np


class PPR_MBFN(nn.Module):
    def __init__(self, feat_dim=512):
        super(PPR_MBFN, self).__init__()
        self.backbone = MobileFacenet(feat_dim=feat_dim)
        
        self.stn = PPR()
        self.m = 0.05
        self.forward_cnt = 0
    
    def forward_train(self, q, g, fp):
        assert q.size() == g.size()
        q_g_fp = torch.cat([q, g, fp], dim=0)
        q_map, g_map, fp_map = self.stn.forward_part1(q_g_fp).split(dim=0, split_size=q.size(0))
        q_g_map_fuse = torch.cat([q_map, g_map], dim=1)
        q_fp_map_fuse = torch.cat([q_map, fp_map], dim=1)
        q_warpto_g = self.stn.forward_part2(q, q_g_map_fuse)
        q_warpto_fp = self.stn.forward_part2(q, q_fp_map_fuse).detach()
        q224, g224, fp224 = self.stn(q), self.stn(g), self.stn(fp)
        q224_g224_qwg224_qwfp224_fp224 = torch.cat(
            [q224, g224, q_warpto_g, q_warpto_fp, fp224], 
            dim=0
        )
        # for the reason of Batch Norm, inference q/g again
        q_feat, g_feat, qwg_feat, qwfp_feat, fp_feat = self.backbone(q224_g224_qwg224_qwfp224_fp224).split(dim=0, split_size=q.size(0))
        tp_metric = F.relu(F.cosine_similarity(q_feat, g_feat).detach() - F.cosine_similarity(qwg_feat, g_feat.detach()) + self.m)
        fp_metric = F.relu(F.cosine_similarity(qwfp_feat, fp_feat) - F.cosine_similarity(q_feat, fp_feat).detach())
        
        feat_all = torch.cat([
            q_feat.unsqueeze(1), 
            qwg_feat.unsqueeze(1), 
            qwfp_feat.unsqueeze(1), 
            g_feat.unsqueeze(1)], dim=1)

        return feat_all, tp_metric, fp_metric

    def forward_test(self, x):
        x = self.stn(x)
        return self.backbone(x)

    def forward_get_224(self, x):
        x224 = self.stn(x)
        return x224, self.backbone(x224)
    
    def forward_test_ft(self, q, g):
        assert q.size() == g.size()
        q_g = torch.cat([q, g], dim=0)
        q_map, g_map = self.stn.forward_part1(q_g).split(dim=0, split_size=q.size(0))
        q_g_map_fuse = torch.cat([q_map, g_map], dim=1)
        q_warpto_g = self.stn.forward_part2(q, q_g_map_fuse)

        return q_warpto_g, self.backbone(q_warpto_g), q_map, g_map

    def forward(self, *args):
        assert len(args) in (1, 2, 3)
        if len(args) == 1:
            return self.forward_test(*args)
        elif len(args) == 2:
            return self.forward_test_ft(*args)
        else:
            return self.forward_train(*args)
        

