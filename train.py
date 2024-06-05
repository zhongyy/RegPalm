import os
import torch.utils.data
from torch import nn
from torch.nn import DataParallel
from datetime import datetime
from core import arcface, mbfn, stn_mbfn, ppr_mbfn
from core.utils import init_log, compute_top1_acc, mat_cosine_similarity
from dataloader.Palm_loader import get_MPD_train_and_test_dataset, get_SelfDefine_train_and_test_dataset
from dataloader.Palm_loader import get_auxiliary_icdm_dataset, get_auxiliary_w270h260_dataset
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
from acctools import roc_tool
from IPython import embed
from collections import OrderedDict
import argparse
from yacs.config import CfgNode

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', type=str, default='', help='path to config file'
    )

    args = parser.parse_args()
    assert os.path.exists(args.config) 
    with open(args.config, "r") as f:
        config = CfgNode.load_cfg(f)

    SAVE_DIR = "./model"
    # gpu init
    gpu_list = ''
    multi_gpus = False
    if isinstance(config.train.gpus, int):
        gpu_list = str(config.train.gpus)
    else:
        multi_gpus = True
        for i, gpu_id in enumerate(config.train.gpus):
            gpu_list += str(gpu_id)
            if i != len(config.train.gpus) - 1:
                gpu_list += ','
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # other init
    start_epoch = 1
    save_dir = os.path.join(
        SAVE_DIR, 
        ".".join(os.path.basename(args.config).split(".")[:-1]) + "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')

    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    gray_scale = False
    random_skew = False if not hasattr(config.data, "random_skew") else config.data.random_skew
    feat_dim=512
    # define model

    if config.model.name == "mbfn":
        feat_dim=512
        net = mbfn.MobileFacenet(feat_dim=feat_dim)
    elif config.model.name == "stnmbfn":
        feat_dim=512
        net = stn_mbfn.STN_MBFN(feat_dim=feat_dim)
    elif config.model.name == "pprmbfn":
        feat_dim=512
        net = ppr_mbfn.PPR_MBFN(feat_dim=feat_dim)
    else:
        raise ValueError("undefined model name {}".format(config.model.name))
    

    if len(config.model.pretrained) == 0:
        print("WARNING: NO pretrained")
    elif config.model.name.startswith("mbfn") or config.model.name.startswith("vismbfn") or config.model.name.startswith("hrmbfn"):
        pretrained_path = config.model.pretrained
        loaded_state_dict = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in loaded_state_dict:
            print("loading imagenet pretrain: ", pretrained_path)
            loaded_state_dict = loaded_state_dict["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in loaded_state_dict.items():
                assert k.startswith("module.")
                new_key = k[len("module."):]
                assert new_key not in loaded_state_dict
                new_state_dict[new_key] = v
           
            if config.model.name.find("f256"):
                for k in [
                    "classifier.weight", "classifier.bias", "linear1.conv.weight", "linear1.bn.weight",
                    "linear1.bn.bias", "linear1.bn.running_mean", "linear1.bn.running_var"
                ]:
                    del new_state_dict[k]

                if hasattr(net, "stn") and len([i for i in new_state_dict if i.startswith("stn.")]) == 0:
                    print("load without stn weights")
                    net.backbone.load_state_dict(new_state_dict, strict=False)
                else:
                    net.load_state_dict(new_state_dict, strict=False)
            else:
                for k in ["classifier.weight", "classifier.bias"]:
                    del new_state_dict[k]

                if hasattr(net, "stn") and len([i for i in new_state_dict if i.startswith("stn.")]) == 0:
                    print("load without stn weights")
                    net.backbone.load_state_dict(new_state_dict, strict=True)
                else:
                    net.load_state_dict(new_state_dict, strict=True)
        else:
            print("loading arcpalm pretrain: ", pretrained_path)
            new_state_dict = loaded_state_dict["net_state_dict"]
            if hasattr(net, "stn") and len([i for i in new_state_dict if i.startswith("stn.")]) == 0:
                print("load without stn weights")
                net.backbone.load_state_dict(new_state_dict, strict=True)
            else:
                net.load_state_dict(new_state_dict, strict=True)
    else:
        raise ValueError("undefined model name {}".format(config.model.name))    


    transform_type = "classic_transform"
    if hasattr(config.data, "transform_type"):
        assert config.data.transform_type in (
            "classic_transform", "lux_transform", "classic_transform_with_overcrop", "lux_transform_with_overcrop"
        )
        transform_type = config.data.transform_type
    elif hasattr(config.data, "lux_transform") and config.data.lux_transform:
        transform_type = "lux_transform"

    if config.data.name == "mpd":
        train_set, query_set, gallery_set = get_MPD_train_and_test_dataset(
            config.data.imglist, 
            transform_type=transform_type,
            fold_idx=config.data.fold_idx, num_folds=config.data.fold_num, 
            gray_scale=gray_scale, random_skew=random_skew,
            use_superclass=config.data.use_superclass if hasattr(config.data, "use_superclass") else False,
            return_2samples=config.data.return_2samples if hasattr(config.data, "return_2samples") else False
        )
    elif config.data.name == "self_define":
        train_set, query_set, gallery_set = get_SelfDefine_train_and_test_dataset(
            config.data.train_list, 
            config.data.test_list, 
            transform_type=transform_type,
            gray_scale=gray_scale, random_skew=random_skew,
            use_superclass=config.data.use_superclass if hasattr(config.data, "use_superclass") else False,
            return_2samples=config.data.return_2samples if hasattr(config.data, "return_2samples") else False
        )
    else:
        raise ValueError("undefined data name {}".format(config.data.name))  

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.train.batch_size,
        shuffle=True, num_workers=8, 
        drop_last=True
    )

    query_loader = torch.utils.data.DataLoader(
        query_set, batch_size=16,
        shuffle=False, num_workers=2, 
        drop_last=False
    )

    gallery_loader = torch.utils.data.DataLoader(
        gallery_set, batch_size=16,
        shuffle=False, num_workers=2, 
        drop_last=False
    )

    aux_loader = None
    if hasattr(config.data, "aux_list"):
        assert hasattr(config.data, "aux_type")
        assert hasattr(config.data, "aux_ratio")
        print("using aux_list:{}, aux_type:{}, aux_ratio:{}".format(config.data.aux_list, config.data.aux_type, config.data.aux_ratio))
        if config.data.aux_type == "icdm":
            aux_set = get_auxiliary_icdm_dataset(
                config.data.aux_list, 
                transform_type=transform_type,
                gray_scale=gray_scale, random_skew=random_skew,
                use_superclass=config.data.use_superclass if hasattr(config.data, "use_superclass") else False,
                return_2samples=config.data.return_2samples if hasattr(config.data, "return_2samples") else False
            )
        elif config.data.aux_type == "w270h260":
            aux_set = get_auxiliary_w270h260_dataset(
                config.data.aux_list,
                transform_type=transform_type,
                gray_scale=gray_scale, random_skew=random_skew,
                use_superclass=config.data.use_superclass if hasattr(config.data, "use_superclass") else False,
                return_2samples=config.data.return_2samples if hasattr(config.data, "return_2samples") else False
            )
        else:
            raise ValueError("undefined aux_type {}".format(config.data.aux_type))  

        aux_batch_size = int(config.train.batch_size * config.data.aux_ratio)
        assert aux_batch_size > 0
        aux_loader = torch.utils.data.DataLoader(
            aux_set, batch_size=aux_batch_size,
            shuffle=True, num_workers=8, 
            drop_last=True
        )

    class_nums = train_set.class_nums if aux_loader is None else train_set.class_nums + aux_set.class_nums
    class_to_superclass = None
    if hasattr(train_set, "class_to_superclass") and train_set.class_to_superclass is not None:
        class_to_superclass = train_set.class_to_superclass
        if aux_loader:
            assert hasattr(aux_set, "class_to_superclass")
            num_base_classes = len(class_to_superclass)
            assert num_base_classes == max([i for i in class_to_superclass]) + 1
            assert num_base_classes == train_set.class_nums
            num_base_super_classes = len(set([i for _, i in class_to_superclass.items()]))
            assert num_base_super_classes == max([i for _, i in class_to_superclass.items()]) + 1
            for k, v in aux_set.class_to_superclass.items():
                class_to_superclass[k + num_base_classes] = v + num_base_super_classes

        assert class_nums == len(class_to_superclass)
        assert class_nums == max([i for i in class_to_superclass]) + 1
        superclass_list = [class_to_superclass[i] for i in range(0, class_nums)]
        print("n_class {}, n_superclassusing {}".format(class_nums, len(set(superclass_list))))    
        superclass_vector = torch.LongTensor(superclass_list)

    ArcMargin = arcface.ArcMarginProduct(
        feat_dim, class_nums,
        class_to_superclass=class_to_superclass
    )

    # define optimizers
    ignored_params = list(map(id, net.linear1.parameters())) if not hasattr(net, "stn") \
        else list(map(id, net.backbone.linear1.parameters()))
    
    ignored_params += list(map(id, ArcMargin.weight))
    prelu_params_id = []
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_ft = optim.SGD([
        {'params': base_params, 'weight_decay': 1e-3},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4} if not hasattr(net, "stn") \
            else {'params': net.backbone.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': ArcMargin.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=config.train.optim.lr_start, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=config.train.optim.steps, gamma=0.1)

    net = net.cuda()
    ArcMargin = ArcMargin.cuda()
    if multi_gpus:
        net = DataParallel(net)
        ArcMargin = DataParallel(ArcMargin)
    criterion = torch.nn.CrossEntropyLoss()

    
    if aux_loader:
        aux_loader_iter = iter(aux_loader)

    n_iter = 0
    for epoch in range(start_epoch, config.train.optim.total_epoch + 1):
        try:
            ArcMargin.update_margin(min(float(epoch) / 20 * 0.5, 0.5))
        except:
            ArcMargin.module.update_margin(min(float(epoch) / 20 * 0.5, 0.5))

        #exp_lr_scheduler.step()
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, config.train.optim.total_epoch))
        net.train()
        ArcMargin.train()

        train_total_loss = 0.0
        total = 0
        since = time.time()
        for data in train_loader:
            if len(data) == 2:
                img, label = data[0], data[1]
                if aux_loader:
                    try:
                        data_aux = next(aux_loader_iter)
                    except:
                        print("reset aux_set loader")
                        aux_loader_iter = iter(aux_loader)
                        data_aux = next(aux_loader_iter)
                    
                    img_aux, label_aux = data_aux[0], data_aux[1]
                    label_aux += train_set.class_nums
                    img, label = torch.cat([img, img_aux], dim=0), torch.cat([label, label_aux], dim=0)
                    idx_shuffle = torch.randperm(img.size(0))
                    img, label = img[idx_shuffle], label[idx_shuffle]
                
                img, label = img.cuda(), label.cuda()
                batch_size = img.size(0)
                optimizer_ft.zero_grad()
                feat = net(img)
                #print(feat.size())
                output = ArcMargin(feat, label)
                total_loss = criterion(output, label)
                total_loss.backward()
                optimizer_ft.step()
            else:
                assert len(data) == 3
                q, g, label = data
                if aux_loader:
                    try:
                        data_aux = next(aux_loader_iter)
                    except:
                        print("reset aux_set loader")
                        aux_loader_iter = iter(aux_loader)
                        data_aux = next(aux_loader_iter)
                    
                    q_aux, g_aux, label_aux = data_aux
                    label_aux += train_set.class_nums
                    q, g, label = torch.cat([q, q_aux], dim=0), torch.cat([g, g_aux], dim=0), torch.cat([label, label_aux], dim=0)
                    idx_shuffle = torch.randperm(q.size(0))
                    q, g, label = q[idx_shuffle], g[idx_shuffle], label[idx_shuffle] 

                super_label = superclass_vector[label.view(-1)]
                q, g, label, super_label = q.cuda(), g.cuda(), label.cuda(), super_label.cuda()

                assert q.size() == g.size()
                
                net.eval()
                with torch.no_grad():
                    q_g = torch.cat([q, g], dim=0)
                    q_g_feat = net(q_g)
                    q_feat, _ = q_g_feat.split(dim=0, split_size=q.size(0))
                    simi = mat_cosine_similarity(q_feat, q_g_feat)
                    q_g_super_label = torch.cat([super_label, super_label], dim=0)
                    same_superlabel_flag = torch.eq(super_label.view(-1, 1), q_g_super_label.view(1, -1)).float()
                    simi += ((-2.0) * same_superlabel_flag)
                    fp_idx = simi.argmax(dim=1)
                    fp = q_g[fp_idx].detach()
                 
                fp.requires_grad_(True)
                net.train()
                batch_size = q.size(0)
                optimizer_ft.zero_grad()
                
                feat, tp_metric, fp_metric = net(q, g, fp)
                label = torch.cat([label, label, label, label], dim=0)
              
                feat = torch.cat(feat.split(dim=1, split_size=1), dim=0).squeeze(1)
                output = ArcMargin(feat, label)
                loss1 = criterion(output, label)
                loss2 = tp_metric.mean()
                loss3 = fp_metric.mean()
                #print(loss1.cpu().item(), loss2.cpu().item(), loss3.cpu().item())

                total_loss = loss1 + loss2 + loss3
                total_loss.backward()
                optimizer_ft.step()

            train_total_loss += total_loss.item() * batch_size
            total += batch_size
            #print(total_loss.cpu().item())
            n_iter += 1
            #if n_iter > 50:
            #    exit()
    
        exp_lr_scheduler.step()
        train_total_loss = train_total_loss / total
        time_elapsed = time.time() - since
        loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s'\
            .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
        _print(loss_msg)

        # test
        if epoch % config.train.test_freq == 0:
            net.eval()
            query_feats = []
            query_labels = []
            label_to_rawlabel = query_set.int_to_label
            with torch.no_grad():
                for img, label in query_loader:
                    img = img.cuda()
                    query_feats.append(net(img).cpu().numpy())
                    query_labels.extend([label_to_rawlabel[i] for i in label.tolist()])

            query_feats = np.vstack(query_feats)

            gallery_feats = []
            gallery_labels = []
            label_to_rawlabel = gallery_set.int_to_label
            with torch.no_grad():
                for img, label in gallery_loader:
                    img = img.cuda()
                    gallery_feats.append(net(img).cpu().numpy())
                    gallery_labels.extend([label_to_rawlabel[i] for i in label.tolist()])

            gallery_feats = np.vstack(gallery_feats)

            if hasattr(config.data, "use_superclass") and config.data.use_superclass is True:
                _print('    use_superclass do not test top1')
                all_labels = query_labels + gallery_labels
                assert all([i.endswith("-l") or i.endswith("-r") for i in all_labels])
                group_labels = [i[-2:] for i in all_labels] 
                roc = roc_tool.roc_curve(np.vstack([query_feats, gallery_feats]), query_labels + gallery_labels, interest_score_min=0.2, device="cuda:0", sample_group=group_labels)
                _print('    tpr@fpr=1e-4: {}'.format(roc.tpr_at_fpr(1e-4)))
                _print('    tpr@fpr=1e-5: {}'.format(roc.tpr_at_fpr(1e-5)))
                _print('    tpr@fpr=1e-6: {}'.format(roc.tpr_at_fpr(1e-6)))
            else:
                top1_acc = compute_top1_acc(query_feats, query_labels, gallery_feats, gallery_labels)
                roc = roc_tool.roc_curve(np.vstack([query_feats, gallery_feats]), query_labels + gallery_labels, interest_score_min=0.2, device="cuda:0")
                _print('    tpr@fpr=1e-4: {}'.format(roc.tpr_at_fpr(1e-4)))
                _print('    tpr@fpr=1e-5: {}'.format(roc.tpr_at_fpr(1e-5)))
                _print('    tpr@fpr=1e-6: {}'.format(roc.tpr_at_fpr(1e-6)))
                _print('    ave top1: {:.4f}'.format(top1_acc * 100))

        # save model
        if epoch % config.train.save_freq == 0:
            msg = 'Saving checkpoint: {}'.format(epoch)
            _print(msg)
            if multi_gpus:
                net_state_dict = net.module.state_dict()
            else:
                net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

    print('finishing training')

if __name__ == '__main__':
    main()
