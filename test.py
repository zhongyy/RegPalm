import argparse
from dataloader.Palm_loader import get_simple_test_dataset
from IPython import embed
from core import mbfn, ppr_mbfn, stn_mbfn
import torch
import numpy as np
from acctools import roc_tool
from tqdm import tqdm
import os
import pickle

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--query', type=str, help='query'
    )
    parser.add_argument(
        '--gallery', type=str, default='', help='gallery'
    )
    parser.add_argument(
        '--fp', type=str, default='', help='fp list'
    )
    parser.add_argument(
        '--model', type=str, default='', help='model name'
    )
    parser.add_argument(
        '--pth', type=str, help='model pth'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', help='device'
    )
    parser.add_argument(
        '--save', type=str, default='', help='save dir'
    )
    parser.add_argument(
        '--group', action="store_true", help='using group (left compare with left)'
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    if len(args.save) > 0:
        assert os.path.exists(args.save)

    gray_scale = False
    transform_type = "classic_transform"
    if args.model == "mbfn":
        net = mbfn.MobileFacenet(feat_dim=512)
    elif args.model.name == "stnmbfn":
        transform_type = "classic_transform_with_overcrop"
        net = stn_mbfn.STN_MBFN(feat_dim=512)
    elif args.model == "pprmbfn":
        transform_type = "classic_transform_with_overcrop"
        net = ppr_mbfn.PPR_MBFN(feat_dim=512)
    else:
        raise ValueError("undefined model name {}".format(args.model))


    net.load_state_dict(torch.load(args.pth, map_location="cpu")["net_state_dict"], strict=True)
    net.eval()
    net.to(device)
    net.eval()

    feat_bank = {}
    
    for feat_name, imglist in (
        ("query", args.query),
        ("gallery", args.gallery),
        ("fp", args.fp),
    ):
        if len(imglist) == 0 or imglist.lower() == "none":
            continue
        
        assert os.path.exists(imglist)
        data_set = get_simple_test_dataset(imglist, gray_scale=gray_scale, transform_type=transform_type)
        data_loader = torch.utils.data.DataLoader(
            data_set, batch_size=4,
            shuffle=False, num_workers=4, drop_last=False
        )
        feats = []
        labels = []
        label_to_rawlabel = data_set.int_to_label
        print("extracting {}: {} ...".format(feat_name, imglist))
        with torch.no_grad():
            for img, label in tqdm(data_loader):
                img = img.to(device)
                feats.append(net(img).cpu().numpy())
                labels.extend([label_to_rawlabel[i] for i in label.tolist()])

        feats = np.vstack(feats)
        feat_bank[feat_name] = (feats, labels)

    assert "query" in feat_bank
    (q_f, q_l)  = feat_bank["query"]
    if "gallery" in feat_bank:
        print("query x gallery ...")
        (g_f, g_l) = feat_bank["gallery"]
    else:
        (g_f, g_l) = None, None
        print("query x query ...")

    assert all([i.endswith("-l") or i.endswith("-r") for i in (q_l+g_l if g_l else q_l)])
    q_g = [i[-2:] for i in q_l] if args.group else None
    g_g = ([i[-2:] for i in g_l] if g_l else None) if args.group else None
    main_roc = roc_tool.roc_curve(q_f, q_l, g_f, g_l, interest_score_min=0.2, device=args.device, 
        sample_group=q_g, template_group=g_g)
    for fp in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        print("  {:.1e}: {}".format(fp, main_roc.tpr_at_fpr(fp)))
        
    roc_bank = {}
    roc_bank["main"] = main_roc
    if "fp" in feat_bank:
        print("using an extra list to set fp table ...")
        (ex_f, ex_l) = feat_bank["fp"]
        assert all([i.endswith("-l") or i.endswith("-r") for i in ex_l])
        ex_g = [i[-2:] for i in ex_l] if args.group else None
        ex_roc = roc_tool.roc_curve(ex_f, ex_l, interest_score_min=0.2, device=args.device,
            sample_group=ex_g)
        for fp in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
            print("  {:.1e}: {}".format(fp, ex_roc.tpr_at_fpr(fp)))
            
        print("tuned query x gallery (or query x query)")
        for fp in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
            score_thresh = ex_roc.tpr_at_fpr(fp)[-1]
            fp_found = ex_roc.tpr_at_fpr(fp)[-2]
            print("  {:.1e}: {:.3f} {:.2e} {:.2f}".format(
                fp,
                main_roc.tpr_at_score(score_thresh), 
                fp_found,
                score_thresh
            ))

        roc_bank["ex"] = ex_roc

    pickle.dump(
        {
            "feat": feat_bank,
            "roc": roc_bank,
        }, 
        open(os.path.join(args.save, "feat_roc.pkl"), 'wb')
    )
    #embed()
       
if __name__ == '__main__':
    main() 
