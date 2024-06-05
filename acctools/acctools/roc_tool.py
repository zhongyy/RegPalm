# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
import time
from IPython import embed
import pickle
import os
from . import util, feat_tool, core
import re
import itertools
from .util import load_roc, load_roc_from_txt, RAND_SEED
from .core import Top3FetcherByScoreFusion, Top3FetcherByFpFusion, \
    cosine_similarity, int8_cosine_similarity

try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest # python2 compatible

def roc_curve(sample_feature, sample_label, template_feature=None, template_label=None, 
        patch_step=2500, 
        device="cpu", 
        similarity_func=cosine_similarity,
        bins=10000, 
        interest_score_min=-1.0, 
        interest_score_max=1.0,
        aggregate_func=core.simple_mean,
        hard_example_visual=None,
        shape_check=True,
        sample_group=None,
        template_group=None,
        enable_debug=False
    ):

    r"""1vs1 ROC curve, an vectorized implementation, compatible with multimodal feats.
    Args:
        sample_feature (np.ndarray, float32, required): query feature (a <list> of or a M*D array)
        sample_label (list<string or int>, required): query label (len: M)
        template_feature (np.ndarray, float32, required): gallery feature (a <list> of or a N*D array)
        template_label (list<string or int>, required): gallery label (len: N)
        bins (int, optional): number of histogram bins
        patch_step (int, optional): the batch size
        device (string, optional): computing device like "cpu", "cuda:0", "cuda:1"
        similarity_func (function, optional): a vectorized function to calculation similarity
        aggregate_func (function, optional): a vectorized function to aggregate multimodal scores
        interest_score_min: scores below this value are directly considered as interclass-compare (for saving time)
        interest_score_max: scores above this value are directly considered as intraclass-compare (for saving time)
    
    Output: 
        util.RocContainer, representing the final roc

    Examples::
        examples/roc/test_roc_1vs1.py
    """

    ########################## pre-treatment ###########################
    with_template = template_feature is not None

    if type(sample_feature) is list:
        if template_feature is not None:
            util.TYPE_CHECK(template_feature, list)
            util.EQ_CHECK(len(sample_feature), len(template_feature), 
                "num modalites for sample", "num modalites for template")
        else:
            assert template_label is None
            template_feature = [None for _ in range(len(sample_feature))]

        if type(similarity_func) is list:
            util.EQ_CHECK(len(sample_feature), len(similarity_func),
                "num modalites for sample", "according similarity_funcs")
        else:
            similarity_func = [similarity_func for _ in range(len(sample_feature))]
    else:
        sample_feature = [sample_feature]
        template_feature = [template_feature]
        similarity_func = [similarity_func]
    
    for fun_id in range(len(similarity_func)):
        similarity_func[fun_id] = core.compatible_simfun(similarity_func[fun_id])

    # check input for each modality
    for modal_idx, ((s_feat, s_label), (t_feat, t_label)) in enumerate( 
            zip_longest(
                itertools.product(sample_feature, [sample_label]), 
                itertools.product(template_feature, [template_label]) ) ):
        util.check_roc_input(
            [ (s_feat, s_label, "query modal_{}".format(modal_idx), True, shape_check),
              (t_feat, t_label, "gallery modal_{}".format(modal_idx), False, shape_check) ] 
        )

    [sample_feature, template_feature] = feat_tool.convert_feats_to_torch(
        list_of_feats=[sample_feature, template_feature] ) 

    sample_label_raw = sample_label
    template_label_raw = template_label
    [sample_label, template_label], _ = feat_tool.convert_labels_to_torch(
        labels=[sample_label, template_label] ) 

    util.TYPE_CHECK(bins, int)
    util.LE_CHECK(bins, 32000, "histo bins", "largest support bins")
    util.GE_CHECK(bins, 100, "histo bins", "least support bins")
    device = torch.device(device)

    # "fp" -> inter-class comparison, "tp" -> intra-class comparison
    ROC_VEC_KEYS = set(["fp", "tp"])
    score_histo_dict = {
        key : torch.zeros(bins, dtype=torch.float64, device=torch.device('cpu'))
        for key in ROC_VEC_KEYS
    }

    VISUAL_KEYS = set(["fp_hard", "tp_hard"])
    if hard_example_visual is not None:
        util.check_visual(hard_example_visual)
        visual_dict = {
            key : util.Bucket(hard_example_visual["volume"])
            for key in VISUAL_KEYS
        }
    else:
        visual_dict = None

    if sample_group is not None or template_group is not None:
        print("using group")
        if with_template:
            assert sample_group is not None and template_group is not None
        else:
            assert sample_group is not None and template_group is None

        util.TYPE_CHECK(sample_group, list)
        util.EQ_CHECK(len(sample_group), len(sample_label_raw), 
                "sample_group len", "sample_label len")
        if with_template:
            util.TYPE_CHECK(template_group, list)
            util.EQ_CHECK(len(template_group), len(template_label), 
                    "template_group len", "template_label len")
    else:
        sample_group = ["group:0" for _ in range(len(sample_label_raw))]
        template_group = ["group:0" for _ in range(len(template_label_raw))] if with_template else None
 
    if with_template:
        util.EQ_CHECK(
            len(set([(i, j) for i, j in zip(
                sample_label_raw + template_label_raw, 
                sample_group + template_group
            )])), 
            len(set(sample_label_raw + template_label_raw)), 
            "label-group set len", 
            "label set len (a label can not belong to more than one groups)"
        )
    else:
        util.EQ_CHECK(
            len(set([(i, j) for i, j in zip(sample_label_raw, sample_group)])), 
            len(set(sample_label_raw)), 
            "label-group set len", 
            "label set len (a label can not belong to more than one groups)"
        )
    
    sample_group_raw = sample_group
    template_group_raw = template_group
    [sample_group, template_group], _ = feat_tool.convert_labels_to_torch(
        labels=[sample_group, template_group] ) 
    
    ########################### main loop #############################
    num_sample = sample_feature[0].size(0)
    if with_template:
        num_template = template_feature[0].size(0)
        print("sample X template Cartesian: {} X {}".format(num_sample, num_template))
        patches = core.get_patches(num_sample, num_template, step=patch_step)
    else:
        # in self vs self comparison, only consider down-triangle patches
        print("sample X sample (itself) Cartesian: {} X {}".format(num_sample, num_sample))
        patches = core.get_patches(num_sample, num_sample, step=patch_step, down_triangle=True)
    
    print("resolved to {} patches".format(len(patches)))

    for patch in tqdm(patches):
        sample_start, sample_end = patch[0]
        template_start, template_end = patch[1]

        sample_label_patch = sample_label[sample_start : sample_end].to(device)
        template_label_patch = \
            template_label[template_start : template_end].to(device) if with_template \
            else sample_label[template_start : template_end].to(device)

        """
        when sample x sample (the are the same feat-vector), only count those scores at 1-position (down-triangle).
        Example:
                1  1  1  1  1  1  1             0  0  0  0  0
                1  1  1  1  1  1  1             1  0  0  0  0
                1  1  1  1  1  1  1             1  1  0  0  0           
                1  1  1  1  1  1  1             1  1  1  0  0 
                1  1  1  1  1  1  1             1  1  1  1  0   
               sample(5) x template(7)      sample(5) x sample(5) 
        """
        valid_ind = torch.ones((sample_end-sample_start, template_end-template_start), 
            dtype=util.torch_comparison_type).to(device) if with_template \
            else core.get_indicate_matrix(
                torch.arange(sample_start, sample_end, dtype=torch.int32).to(device), 
                torch.arange(template_start, template_end, dtype=torch.int32).to(device), 
                OP="GT")

        sample_group_patch = sample_group[sample_start : sample_end].to(device)
        template_group_patch = \
            template_group[template_start : template_end].to(device) if with_template \
            else sample_group[template_start : template_end].to(device)

        """
        Count on groups, only count positions where sample and template are in the same group.
            And in default, sample and template all belong to one unique group.
        Example:
            template groups
                     2  1  1  3  2
        sample groups
                1    0  1  1  0  0
                1    0  1  1  0  0
                2    1  0  0  0  1      
                4    0  0  0  0  0
                
               sample(4) x template(5) 
        """
        same_group_ind = core.get_indicate_matrix(sample_group_patch, template_group_patch, OP="EQ")
        valid_ind *= same_group_ind
            
        scores = []
        for sample_feat_modal, template_feat_modal, sim_fun in zip_longest(
                sample_feature, template_feature, similarity_func):
            s_feat = sample_feat_modal[sample_start : sample_end].to(device)
            t_feat = template_feat_modal[template_start : template_end].to(device) if with_template \
                else sample_feat_modal[template_start : template_end].to(device)
            
            score = sim_fun(s_feat, t_feat) 
            scores.append(score)

        final_score = aggregate_func(scores)
        assert final_score.max().item() < 1.0 and final_score.min().item() > -1.0, \
            "make sure score in (-1, 1), got (min:{}, max:{})".format(
                final_score.min().item(), final_score.max().item()
            )

        underflow_ind = torch.lt(final_score, interest_score_min)
        overflow_ind = torch.gt(final_score, interest_score_max)
        interest_ind = torch.mul(~underflow_ind, ~overflow_ind)

        pos_ind = core.get_indicate_matrix(sample_label_patch, template_label_patch, OP="EQ")
        score_masks = {
            "fp"  : valid_ind * interest_ind * (~pos_ind),
            "tp"  : valid_ind * interest_ind * ( pos_ind),
        }
        extra_masks = {
            "underflow_fp" : valid_ind * underflow_ind * (~pos_ind),
            "underflow_tp" : valid_ind * underflow_ind * ( pos_ind),
            "overflow_fp"  : valid_ind * overflow_ind  * (~pos_ind),
            "overflow_tp"  : valid_ind * overflow_ind  * ( pos_ind),
        }

        util.EQ_CHECK(ROC_VEC_KEYS, set(score_masks.keys()))
        for key in score_masks:
            score_histo_dict[key] += core.mask_histc(
                src=final_score.view(-1), 
                mask=score_masks[key].view(-1), 
                bins=bins, min_val=-1, max_val=1)

        score_histo_dict["fp"][0] += extra_masks["underflow_fp"].long().sum().cpu().item()
        score_histo_dict["tp"][0] += extra_masks["underflow_tp"].long().sum().cpu().item()
        score_histo_dict["fp"][-1] += extra_masks["overflow_fp"].long().sum().cpu().item()
        score_histo_dict["tp"][-1] += extra_masks["overflow_tp"].long().sum().cpu().item()

        if hard_example_visual is not None:
            sample_idx = torch.arange(sample_start, sample_end, dtype=torch.int32).to(device)
            template_idx = torch.arange(template_start, template_end, dtype=torch.int32).to(device)
            visual_case_masks = {
                "fp_hard" : score_masks["fp"] * torch.ge(final_score, hard_example_visual["fp_thres"]),
                "tp_hard" : score_masks["tp"] * torch.le(final_score, hard_example_visual["tp_thres"]),
            }
            util.EQ_CHECK(VISUAL_KEYS, set(visual_case_masks.keys()))
            for key in visual_case_masks:
                hard_sample_idx, hard_template_idx = visual_case_masks[key].nonzero().unbind(dim=1)
                if hard_sample_idx.size(0) > 0:
                    visual_item = zip(
                        sample_idx[hard_sample_idx].cpu().numpy().tolist(),
                        template_idx[hard_template_idx].cpu().numpy().tolist(),
                        final_score[[hard_sample_idx, hard_template_idx]].cpu().numpy().tolist(),
                    )
                    visual_dict[key].add( [i for i in visual_item] )


    ########################### convert ################################
    score_cumulate_dict = {
        key : torch.cumsum(input=score_histo_dict[key], dim=0) 
        for key in score_histo_dict
    }
    score_count_dict = {
        key: round(score_cumulate_dict[key].max().item())
        for key in score_cumulate_dict
    }

    ########################### numerical check #########################
    label_all = set(sample_label_raw + template_label_raw) if with_template else set(sample_label_raw)
    label_count = {
        "sample" : { i : 0 for i in label_all},
        "template" : { i : 0 for i in label_all},
    }
    for label in sample_label_raw:
        label_count["sample"][label] += 1

    if with_template:
        for label in template_label_raw:
            label_count["template"][label] += 1

    label_to_group = {
        label : group 
        for label, group in zip(
            sample_label_raw + template_label_raw if with_template else sample_label_raw, 
            sample_group_raw + template_group_raw if with_template else sample_group_raw)
    } # each label has already been guaranteed not to map to more than one group

    group_count = {
        "sample": {
            i : len([j for j in sample_label_raw if label_to_group[j] == i]) 
            for i in set(sample_group_raw)
        }
    }
    if with_template:
        group_count["template"] = {
            i : len([j for j in template_label_raw if label_to_group[j] == i]) 
            for i in set(sample_group_raw + template_group_raw)
        }

    # total num for inter-class comparisons
    num_negative_compare = \
        sum( [ label_count["sample"][i] * (group_count["template"][label_to_group[i]] - label_count["template"][i]) 
                for i in label_all ] ) if with_template \
        else sum( [ label_count["sample"][i] * (group_count["sample"][label_to_group[i]] - label_count["sample"][i]) 
                    for i in label_all ] ) / 2 

    # total num for intra-class comparisons
    num_positive_compare = \
        sum( [ label_count["sample"][i] * label_count["template"][i]
                for i in label_all ] ) if with_template \
        else sum( [ label_count["sample"][i] * (label_count["sample"][i] - 1)
                    for i in label_all ] ) / 2 
    
    num_total_compare = sum([
        sample_gp_num * group_count["template"][gp] if with_template \
            else sample_gp_num * (sample_gp_num - 1) / 2
        for gp, sample_gp_num in group_count["sample"].items()
    ])

    util.EQ_CHECK(num_negative_compare + num_positive_compare, num_total_compare)
    util.EQ_CHECK(score_count_dict["fp"], num_negative_compare)
    util.EQ_CHECK(score_count_dict["tp"], num_positive_compare)
    ########################################################################

    vector_dict = {
        # float64 is only useful in counting, convert back to float32 for saving space
        key : torch.div( 
                score_cumulate_dict[key].max() - score_cumulate_dict[key], 
                score_cumulate_dict[key].max()
            ).to(torch.float32).numpy()
        for key in score_cumulate_dict
    }

    num_template = len(template_label_raw) if template_label_raw else 0
    meta = np.array([len(sample_label_raw), num_template], dtype=np.int32)

    if enable_debug:
        return score_histo_dict, meta

    return util.RocContainer(vector_dict, meta=meta, buckets=visual_dict) 


def roc_curve_1vsN(sample_feature, sample_label, template_feature, template_label, 
        patch_step=200,  
        device="cpu",
        similarity_func=cosine_similarity,
        N=None,
        repeats=1,
        max_outlie_ratio=10.0,
        risk_control=None,
        hard_example_visual=None,
        sample_flag=None,
        template_flag=None,
        top3_fetcher=None,
        bins=10000,
        shape_check=True):
    
    r"""1vsN ROC curve. An vectorized implementation.
    Args:
        sample_feature (np.ndarray, float32, required): query feature (shape: m * feat_dim)
        sample_label (list<string or int>, required): query label (len: m)
        template_feature (np.ndarray, float32, required): gallery feature (shape: n * feat_dim)
        template_label (list<string or int>, required): gallery label (len: m)
        bins (int, optional): number of histogram bins
        patch_step: the batch size
        N (int, optional): simulate gallery size
        device (string, optional): computing device like "cpu", "cuda:0", "cuda:1"
        sample_flag (np.ndarray, bool, optional): \
            Set i-th to be False to represent i-th sample is absent (len: m). \
            Each 'sample_feature' (or 'modality') should be given a 'sample_flag'.
        template_flag (np.ndarray, bool, optional):  \
            Set i-th to be False to represent i-th template is absent (len: n). \
            Each 'template_feature' (or 'modality') should be given a 'template_flag'.
    
    Output: 
        list<util.RocContainer>, each representing a roc with specific N (gallery size) 

    Examples::
        examples/roc/test_roc_1vsN.py
    """    
    ############################## setup ####################################
    if top3_fetcher is None:
        top3_fetcher = Top3FetcherByScoreFusion()
    else:
        assert isinstance(top3_fetcher, core.Top3FetcherBase)
        
    if type(sample_feature) is list:
        util.TYPE_CHECK(template_feature, list)
        util.EQ_CHECK(len(sample_feature), len(template_feature), 
                "num modalites for query", "num modalites for gallery")

        if type(similarity_func) is list:
            util.EQ_CHECK(len(sample_feature), len(similarity_func),
                "num modalites for sample", "according similarity_funcs")
        else:
            similarity_func = [similarity_func for _ in range(len(sample_feature))]
    else:
        sample_feature = [sample_feature]
        template_feature = [template_feature]
        similarity_func = [similarity_func]

    for fun_id in range(len(similarity_func)):
        similarity_func[fun_id] = core.compatible_simfun(similarity_func[fun_id])

    # check input for each modality
    for modal_idx, ((s_feat, s_label), (t_feat, t_label)) in enumerate( 
            zip_longest(
                itertools.product(sample_feature, [sample_label]), 
                itertools.product(template_feature, [template_label]) ) ):
        util.check_roc_input(
            [ (s_feat, s_label, "query modal_{}".format(modal_idx), True, shape_check),
              (t_feat, t_label, "gallery modal_{}".format(modal_idx), True, shape_check) ] 
        )
    
    # template has to be distinct
    util.EQ_CHECK(
        len(template_label), len(set(template_label)), 
        "num of template labels", "num of distinct template labels" )

    [sample_feature, template_feature] = feat_tool.convert_feats_to_torch(
        list_of_feats=[sample_feature, template_feature] ) 

    [sample_label, template_label], _ = feat_tool.convert_labels_to_torch(
        labels=[sample_label, template_label] ) 
  
    num_modal = len(sample_feature)
    num_template = template_label.size(0)
    num_sample = sample_label.size(0)

    sample_flag = [np.ones(num_sample, dtype=np.bool) for _ in range(num_modal)] \
        if sample_flag is None else sample_flag
    template_flag = [np.ones(num_template, dtype=np.bool) for _ in range(num_modal)] \
        if template_flag is None else template_flag
    
    for flags, siz in [(sample_flag, num_sample), (template_flag, num_template)]:
        util.TYPE_CHECK(flags, list)
        util.EQ_CHECK(len(flags), num_modal)
        for flag in flags:
            util.TYPE_CHECK(flag, np.ndarray)
            util.EQ_CHECK(flag.dtype, np.bool)
            util.EQ_CHECK(flag.ndim, 1)
            util.EQ_CHECK(flag.shape[0], siz)

    sample_flag = [torch.from_numpy(flag.astype(np.uint8)).to(util.torch_comparison_type) for flag in sample_flag]
    template_flag = [torch.from_numpy(flag.astype(np.uint8)).to(util.torch_comparison_type) for flag in template_flag]
    
    assert type(bins) is int, "type error, {} vs int".format(type(bins))
    assert bins <= 20000 and bins >= 100, "bins shouled be between 100 and 20000, default 10000"
    assert patch_step <= 2000, "patch_step don't need to be very large, should be less than 2000"
    device = torch.device(device)
    ########################## risk control config ###########################   
    RISK_CTL_KEYS = ["delta_top1_top2", "c1", "c2", "c3", "c4", "b", "thres"]
    if risk_control is not None:
        for key in RISK_CTL_KEYS:
            assert key in risk_control, "risk control requires key: {}".format(key)
        
        print("Enable Risk-control")
        print("Risk-Control setting: {}".format(risk_control))
    ############################ gallery configuration #######################
    N = num_template if N is None else N
    util.LE_CHECK(N, num_template, "N to simulate 1:N", "the whole gallery where N is sampled from")
    assert N <= 1100000, "the N is too big({}), are u sure it's your (1:N) gallery?".format(N)
    assert N >= 10, "the N is too small({}), don't u think should it be at least 10?".format(N)

    galleries = util.generate_gallery([N], num_template, repeats=repeats)

    ROC_VEC_KEYS = set([
        "fp", "fp_loose", "tp", "tp_strict", "out", "out_loose", ])
    VISUAL_KEYS = set([
        "fp_hard", "out_hard", "tp_hard"])

    for gallery_idx, the_gallery in enumerate(galleries):
        gallery_indexes = the_gallery["index"]
        template_label_patch = template_label[torch.from_numpy(gallery_indexes)]
        ####### score histogram #####
        the_gallery["score_histo"] = {
            key : torch.zeros(bins, dtype=torch.float64, device=torch.device('cpu'))
            for key in ROC_VEC_KEYS
        }
        ######### visual ############
        if hard_example_visual is not None:
            util.check_visual(hard_example_visual)
            the_gallery["buckets"] = {
                key : util.Bucket(hard_example_visual["volume"])
                for key in VISUAL_KEYS
            }
        #############################

    ###########################################################################
    # the main loop starts !
    #tiktoker = util.Tiktok()
    sample_label_list = sample_label.cpu().numpy().tolist()
    np.random.seed(RAND_SEED)
    for gallery_idx, the_gallery in enumerate(galleries):
        gallery_indexes = the_gallery["index"]
        template_label_patch = template_label[torch.from_numpy(gallery_indexes)].to(device)
        template_feature_patch = [
            t_feat[torch.from_numpy(gallery_indexes)].to(device)
            for t_feat in template_feature
        ]
        template_flag_patch = [
            flag[torch.from_numpy(gallery_indexes)].to(device)
            for flag in template_flag
        ]

        template_label_patch_dict = {
            i : 0 for i in template_label_patch.cpu().numpy().tolist() }
        sample_inlie_indexes = np.array(
            [i for i, label in enumerate(sample_label_list) if label in template_label_patch_dict],
            dtype=np.int64
        )
        sample_outlie_indexes = np.array(
            [i for i, label in enumerate(sample_label_list) if label not in template_label_patch_dict],
            dtype=np.int64
        )
        np.random.shuffle(sample_outlie_indexes)
        sample_outlie_indexes = sample_outlie_indexes[ : int(len(sample_inlie_indexes) * max_outlie_ratio)]

        out_ind_cat = torch.cat([
            torch.zeros(sample_inlie_indexes.shape[0], dtype=util.torch_comparison_type),
            torch.ones(sample_outlie_indexes.shape[0], dtype=util.torch_comparison_type) ], dim=0)

        sample_indexes_cat = torch.cat([
            torch.from_numpy(idxes) 
            for idxes in (sample_inlie_indexes, sample_outlie_indexes) ], dim=0)

        # query set could be very big, so we have to read it patch by patch
        # for example, when num_sample=101, patch_step=50, it gives [(0, 50), (50, 100), (100, 101)]
        sample_patches = [ (i, min(sample_indexes_cat.size(0), i + patch_step))
            for i in range(0, sample_indexes_cat.size(0), patch_step)
        ]

        print("Process gallery {}/{} {}x{}".format(
            gallery_idx + 1, 
            len(galleries), 
            sample_indexes_cat.size(0), 
            template_label_patch.size(0) ) )
        
        for patch in tqdm(sample_patches):
            #tiktoker.tik()

            sample_start, sample_end = patch
            sample_indexes_patch = sample_indexes_cat[sample_start : sample_end]
            sample_feature_patch = [
                s_feat[sample_indexes_patch].to(device)
                for s_feat in sample_feature
            ]
            sample_flag_patch = [
                flag[sample_indexes_patch].to(device)
                for flag in sample_flag
            ]
            sample_label_patch = sample_label[sample_indexes_patch].to(device)
            out_ind = out_ind_cat[sample_start : sample_end].to(device)
            #tiktoker.tok("sample prepare")

            top3_scores, top3_inds = top3_fetcher.fetch(
                sample_feature_patch, 
                template_feature_patch, 
                sample_flag_patch, 
                template_flag_patch,
                similarity_func
            )
            #tiktoker.tok("topk")
            
            top1_score = top3_scores[:,0] 
            top1_ind = top3_inds[:,0]
            top2_score = top3_scores[:,1] 
            top2_ind = top3_inds[:,1]
            top3_score = top3_scores[:,2] 

            top1_label = template_label_patch[top1_ind]
            # truely identify - top1 is me!
            true_ind = torch.eq(top1_label, sample_label_patch)

            top2_label = template_label_patch[top2_ind]
            # top2's label and top1's label must not be the same
            assert torch.eq(top1_label, top2_label).max().int() <= 0

            #tiktoker.tok("topk")

            if risk_control:
                ########################## risk control (1) ###########################
                # top1 - top2 <= delta -> risk!
                top1_risk_ind = torch.le(top1_score - top2_score, risk_control["delta_top1_top2"])
                top2_risk_ind = torch.le(top2_score - top3_score, risk_control["delta_top1_top2"])
                ########################## risk control (2) ###########################
                # prediction_score > LR_thres -> risk!
                # re-scale score from (-1, 1) to (0, 1)
                verify_sim_score = ((top1_score + 1.0) / 2)
                link_sim_score = ((top2_score + 1.0) / 2)
                d_value = verify_sim_score - link_sim_score
                abs_gap = torch.abs(d_value)

                top1_risk_ind += torch.gt(
                    1.0 / (1 + torch.exp( 
                        (   verify_sim_score * risk_control["c1"] + 
                            link_sim_score * risk_control["c2"] + 
                            d_value * risk_control["c3"] + 
                            abs_gap * risk_control["c4"] + 
                            risk_control["b"]
                        ) * (-1.0)
                    )),
                    risk_control["thres"]
                )

                verify_sim_score = ((top2_score + 1.0) / 2)
                link_sim_score = ((top3_score + 1.0) / 2)
                d_value = verify_sim_score - link_sim_score
                abs_gap = torch.abs(d_value)

                top2_risk_ind += torch.gt(
                    1.0 / (1 + torch.exp( 
                        (   verify_sim_score * risk_control["c1"] + 
                            link_sim_score * risk_control["c2"] + 
                            d_value * risk_control["c3"] + 
                            abs_gap * risk_control["c4"] + 
                            risk_control["b"]
                        ) * (-1.0)
                    )),
                    risk_control["thres"]
                )
            else:
                top1_risk_ind = torch.zeros(sample_end - sample_start, dtype=util.torch_comparison_type).to(device)
                top2_risk_ind = torch.zeros(sample_end - sample_start, dtype=util.torch_comparison_type).to(device)
                # no risk control

            #tiktoker.tok("risk control")

            #######################################################################
            top1_score_masks = {
                "fp_loose"   : (~out_ind) * (~true_ind)                    ,
                "fp"         : (~out_ind) * (~true_ind) * (~top1_risk_ind) ,
                "tp"         : (~out_ind) * ( true_ind)                    ,
                "tp_strict"  : (~out_ind) * ( true_ind) * (~top1_risk_ind) ,
                "out_loose"  : ( out_ind)                                  ,
                "out"        : ( out_ind) *               (~top1_risk_ind) ,
            }
            
            top2_score_masks = {
                "out_loose"  : (~out_ind) * ( true_ind)                    ,
                "out"        : (~out_ind) * ( true_ind) * (~top2_risk_ind) ,
            }
            #tiktoker.tok("various masks")

            util.EQ_CHECK(ROC_VEC_KEYS, set(top1_score_masks.keys()))
            for key in top1_score_masks:
                the_gallery["score_histo"][key] += core.mask_histc(src=top1_score, mask=top1_score_masks[key], 
                    bins=bins, min_val=-1, max_val=1)

            for key in top2_score_masks:
                the_gallery["score_histo"][key] += core.mask_histc(src=top2_score, mask=top2_score_masks[key], 
                    bins=bins, min_val=-1, max_val=1)

            #tiktoker.tok("histc")

            if hard_example_visual is not None:
                visual_case_masks = {
                    "fp_hard" : top1_score_masks["fp"] * torch.ge(top1_score, hard_example_visual["fp_thres"]),
                    "out_hard" : top1_score_masks["out"] * torch.ge(top1_score, hard_example_visual["fp_thres"]),
                    "tp_hard" : top1_score_masks["tp"] * torch.le(top1_score, hard_example_visual["tp_thres"]),
                }
                util.EQ_CHECK(VISUAL_KEYS, set(visual_case_masks.keys()))
                for key in visual_case_masks:
                    mask_idx = visual_case_masks[key].nonzero().unbind(dim=1)[0]
                    if mask_idx.size(0) > 0:
                        visual_item = zip(
                            sample_indexes_patch[mask_idx.cpu()].numpy().tolist(),
                            top1_score[mask_idx].cpu().numpy().tolist(),
                            top2_score[mask_idx].cpu().numpy().tolist(),
                            gallery_indexes[top1_ind[mask_idx].cpu().numpy()].tolist(),
                            gallery_indexes[top2_ind[mask_idx].cpu().numpy()].tolist(),
                        )
                        #visual_dict[key].add( [i for i in visual_item] )
                        the_gallery["buckets"][key].add( [i for i in visual_item] )

            #tiktoker.tok("visual")

        # continue gallery loop
        score_cumulate_dict = {
            key : torch.cumsum(input=the_gallery["score_histo"][key], dim=0) 
            for key in the_gallery["score_histo"]
        }
        score_count_dict = {
            key: round(score_cumulate_dict[key].max().item())
            for key in score_cumulate_dict
        } 
        ########################### numerical check ############################
        # total num for "I'm in the gallery"
        num_inlie_sample = round((~out_ind_cat).to(torch.float64).sum().item())
        # total num for "I'm not in the gallery"
        num_outlie_sample = round(out_ind_cat.to(torch.float64).sum().item())

        util.EQ_CHECK(score_count_dict["fp_loose"] + score_count_dict["tp"], num_inlie_sample)
        util.EQ_CHECK(score_count_dict["out_loose"], num_outlie_sample + score_count_dict["tp"])
        ########################################################################
        print( "N={}, In={}({} TP, {} FP), Out={}({} raw + {} pseu, {} pass)".format( 
                N, num_inlie_sample, score_count_dict["tp"], score_count_dict["fp"],
                score_count_dict["out_loose"], num_outlie_sample, score_count_dict["tp"],
                score_count_dict["out"]) )

        dominator_dict = {
            "fp_loose"   : num_inlie_sample,
            "fp"         : num_inlie_sample,
            "tp"         : num_inlie_sample,
            "tp_strict"  : num_inlie_sample,
            "out_loose"  : score_count_dict["out_loose"],
            "out"        : score_count_dict["out_loose"],
        }
        util.EQ_CHECK( set(dominator_dict),  set(score_cumulate_dict))
        util.EQ_CHECK( set(score_count_dict),  set(score_cumulate_dict))
        vector_dict = {
            # float64 is only useful in counting, convert back to float32 for saving space
            key : torch.div( 
                score_cumulate_dict[key].max() - score_cumulate_dict[key], 
                dominator_dict[key]
            ).to(torch.float32).numpy()
            for key in dominator_dict
        }
        meta_vector = np.array([N, num_inlie_sample, num_outlie_sample], dtype=np.int32)
        buckets = the_gallery["buckets"] if hard_example_visual is not None else None
        the_gallery["roc"] = util.RocContainer(vectors=vector_dict, meta=meta_vector, buckets=buckets)

    # merge rocs
    rocs = [i["roc"] for i in galleries]
    return util.merge_rocs(rocs)


def topk_pre(query_feat, query_label, base_feat, base_label):
    mat = cosine_similarity(query_feat, base_feat)
    ind = core.get_indicate_matrix(query_label, base_label, OP="EQ")
    return mat, ind

def topk(query_feat, query_label, base_feat, base_label, k):
    mat, ind = topk_pre(query_feat, query_label, base_feat, base_label)
    if type(k) is list:
        result = []
        for i in k:
            topk = torch.gather(ind, 1, mat.topk(k=i)[1])
            rank_k_acc = topk.sum(dim=1).nonzero().size(0) / float(topk.size(0))
            result.append(rank_k_acc)

        return result 
    else:
        topk = torch.gather(ind, 1, mat.topk(k=k)[1])
        rank_k_acc = topk.sum(dim=1).nonzero().size(0) / float(topk.size(0))
        return rank_k_acc

def topk_badcase(query_feat, query_label, base_feat, base_label, k, num_target=3):
    r"""
    Return bad cases that don't hit the target in top-k. Return three lists,
        the first list is
        [0 , 2, 51, ...] the indexes of problematic query
        the second list is
        [
            ((wrong_top1_index, wrong_top2_index, ...), (wrong_top1_score, wrong_top1_score2, ...)),
            ...
        ] 
        the first list is
        [
            ((true_target1_index, true_target2_index, ...), (true_score_1, true_score_2, ...)),
            ...
        ] 
    Arguments:
        num_target : (int, optional) the max num of true targets to show 
            if num_target=3, (20,21) will represent two-exsited ground true target 
            (20th, 21th), and there is no third target to be found.
    """
    mat, ind = topk_pre(query_feat, query_label, base_feat, base_label)
    topk_score, topk_ind = mat.topk(k=k)
    # "torch.gather(mat, 1, topk_ind)" is identical to "topk_score"
    topk_hit_num = torch.gather(ind, 1, topk_ind).sum(dim=1)
    # bad indexes [0, 2, 51, ...], 1-D
    topk_unhit = torch.lt(topk_hit_num, 1).nonzero().view(-1)
    # len(topk_unhit) x k, 2-D, scores
    topk_wrong_score = topk_score.index_select(0, topk_unhit)
    # len(topk_unhit) x k, 2-D, indexes
    topk_wrong_ind = topk_ind.index_select(0, topk_unhit)

    wrong_topk_index_scores = [
        (tuple(topk_wrong_ind[i].numpy()), tuple(topk_wrong_score[i].numpy()))
        for i in range(0, topk_wrong_ind.size(0))
    ]

    true_target_index_scores = []
    for i in tqdm(topk_unhit):
        true_target_indexes = ind[i].nonzero().view(-1).numpy()
        np.random.shuffle(true_target_indexes)
        true_target_indexes = true_target_indexes[: min(num_target, len(true_target_indexes))]
        true_target_scores = mat[i].index_select(0, torch.from_numpy(true_target_indexes))
        true_target_index_scores.append((tuple(true_target_indexes), tuple(true_target_scores.numpy())))
    
    return [i for i in topk_unhit], wrong_topk_index_scores, true_target_index_scores

def mAP(query_feat, query_label, base_feat, base_label):
    r"""Perform mAP in ReID. 
    """
    mat, ind = topk_pre(query_feat, query_label, base_feat, base_label)

    """
    mat:
    tensor([[1,3,-2,8,2],
            [0,-2,3,4,1]])
    ind:
    tensor([[0,1,1,0,0],
            [1,0,0,1,1]])
    """
    # sort in dim=1
    _, topk_ind = mat.topk(k=mat.size(1))
    topk_ind_gather = torch.gather(ind, 1, topk_ind)

    """
    topk_ind_gather:
    tensor([[0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0]])
    """
    precision = topk_ind_gather.cumsum(dim=1).float() / torch.arange(1, 1+mat.size(1), dtype=torch.float32)  
    """
    precision:
    tensor([[0.0000, 0.5000, 0.3333, 0.2500, 0.4000],
            [1.0000, 0.5000, 0.6667, 0.7500, 0.6000]])
    """
    precision_masked = torch.mul(precision, topk_ind_gather.float())
    average_precision = precision_masked.sum(dim=1) / topk_ind_gather.sum(dim=1).float()
    mean_average_precision = average_precision.mean()
    return mean_average_precision

def roc_curve_binarycls(predprobs, labels, bins=10000, hard_example_visual=None):
    r"""Perform roc curve in sklearn style. 
    """
    util.TYPE_CHECK(predprobs, np.ndarray)
    util.EQ_CHECK(predprobs.dtype, np.float32)
    util.EQ_CHECK(predprobs.ndim, 1)
    util.TYPE_CHECK(labels, np.ndarray)
    util.EQ_CHECK(labels.dtype, bool)
    util.EQ_CHECK(labels.ndim, 1)
    util.EQ_CHECK(len(predprobs), len(labels))

    predprobs = torch.from_numpy(predprobs)
    labels = torch.from_numpy(labels).to(util.torch_comparison_type)
    assert predprobs.max().item() < 1.0001 and predprobs.min().item() > -0.0001, \
            "make sure score in (-1, 1), got (min:{}, max:{})".format(
                predprobs.min().item(), predprobs.max().item()
            )
 
    VISUAL_KEYS = set(["fp_hard", "tp_hard"])
    if hard_example_visual is not None:
        util.check_visual(hard_example_visual)
        visual_dict = {
            key : util.Bucket(hard_example_visual["volume"])
            for key in VISUAL_KEYS
        }
    else:
        visual_dict = None

    #pos_ind = core.get_indicate_matrix(sample_label_patch, template_label_patch, OP="EQ")
    
    ROC_VEC_KEYS = set(["fp", "tp"])
    score_histo_dict = {
        key : torch.zeros(bins, dtype=torch.float64, device=torch.device('cpu'))
        for key in ROC_VEC_KEYS
    }
    score_masks = {
        "fp"  : (~labels),
        "tp"  : labels,
    }
    util.EQ_CHECK(ROC_VEC_KEYS, set(score_masks.keys()))
    score_histo_dict = {
        key : core.mask_histc(
            src=predprobs.view(-1), 
            mask=score_masks[key].view(-1), 
            bins=bins, min_val=0, max_val=1)
        for key in ROC_VEC_KEYS
    }

    if hard_example_visual is not None:
        sample_idx = torch.arange(0, len(predprobs), dtype=torch.int32)
        visual_case_masks = {
            "fp_hard" : score_masks["fp"] * torch.ge(predprobs, hard_example_visual["fp_thres"]),
            "tp_hard" : score_masks["tp"] * torch.le(predprobs, hard_example_visual["tp_thres"]),
        }
        util.EQ_CHECK(VISUAL_KEYS, set(visual_case_masks.keys()))
        for key in visual_case_masks:
            hard_sample_idx = visual_case_masks[key].nonzero().view(-1)
            if hard_sample_idx.size(0) > 0:
                visual_item = zip(
                    sample_idx[hard_sample_idx].cpu().numpy().tolist(),
                    predprobs[hard_sample_idx].cpu().numpy().tolist(),
                )
                visual_dict[key].add( [i for i in visual_item] )

    score_cumulate_dict = {
        key : torch.cumsum(input=score_histo_dict[key], dim=0) 
        for key in score_histo_dict
    }

    vector_dict = {
        # float64 is only useful in counting, convert back to float32 for saving space
        key : torch.div( 
                score_cumulate_dict[key].max() - score_cumulate_dict[key], 
                score_cumulate_dict[key].max()
            ).to(torch.float32).numpy()
        for key in score_cumulate_dict
    }
    
    meta = np.array([len(predprobs)], dtype=np.int32)
    return util.RocContainer(vector_dict, meta=meta, buckets=visual_dict, min_val=0.0, max_val=1.0) 