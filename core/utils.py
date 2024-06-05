from __future__ import print_function
import os
import logging
from acctools import feat_tool, roc_tool
import torch
import numpy as np

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def compute_top1_acc(query_feat, query_label, base_feat, base_label):
    [query_feat, base_feat] = feat_tool.convert_feats_to_torch(
        list_of_feats=[[query_feat], [base_feat]]
    ) 
    query_feat = query_feat[0]
    base_feat = base_feat[0]
    [query_label, base_label], _ = feat_tool.convert_labels_to_torch(
        labels=[query_label, base_label] 
    ) 
    return roc_tool.topk(query_feat, query_label, base_feat, base_label, k=1)

def mat_cosine_similarity(a, b):
    matmul = torch.matmul(a, b.t())
    a_norm_sqrt = torch.sqrt(torch.sum(torch.mul(a, a), 1, keepdim=True))
    b_norm_sqrt = torch.sqrt(torch.sum(torch.mul(b, b), 1, keepdim=True))
    norm_sqrt_mul = torch.matmul(a_norm_sqrt, b_norm_sqrt.t())
    return torch.clamp(torch.div(matmul, norm_sqrt_mul), -0.99999, 0.99999)

def save_tensor_disk(x, name):
    return
    
    save_dir = os.path.join("./test_cache", "saved_tentor")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    np.save(
        os.path.join(save_dir, "{}_index-{}.npy".format(name, x.device.index)), 
        x.detach().cpu().numpy()
    )
    return


if __name__ == '__main__':
    pass
