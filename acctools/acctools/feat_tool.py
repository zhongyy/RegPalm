#######
# filename: feat_tool.py
# author: junan
# version: 0.1
#######

import numpy as np
from tqdm import tqdm
#from IPython import embed
import os
import torch
import re
from . import util

def str_substitute(inp, chars):
    if inp in chars:
        return " "
    else:
        return inp

def multi_split(str_in, split_chars=[","]):
    for i in split_chars:
        assert len(i) == 1, "only support single-char, got{}".format(i)
    
    str_in = "".join([str_substitute(i, split_chars) for i in str_in])
    return str_in.split()

def np_split(str_in, split_chars=[" "]):
    for i in split_chars:
        str_in = np.char.replace(str_in, i, " ")
    
    return str(str_in).split()

def read_feat_from_txt(txt, label_idx, feat_start_idx, feat_end_idx=None, split_chars=[" "]):
    r""" This function reads features from txt file.
    It read features that are wrote as following formats:
        10001 0.22 0.001 -0.51 ... 0.33
        0.22 0.001 -0.51 ... 0.33 10001  (where 10001 is label)
        img.jpg 0 0 50 50 0.22 0.001 -0.51 ... (where img.jpg is label, 0 0 50 50 are useless)

    Arguments:
        txt (string, required): path to txt file
        label_idx (int, required): the index that indicate label
        feat_start_idx (int, required): the index where feature starts (included)
        feat_end_idx (int, optional): the index where features ends (excluded)
    """

    assert(type(split_chars) is list)
    util.GT_CHECK(len(split_chars), 0)
    for i in split_chars:
        assert len(i) == 1, "only support single-char, got{}".format(i)

    labels = []
    feats = []

    with open(txt, "r") as txt_file:
        initialized = False
        while True:
            one_line = txt_file.readline()
            if not one_line:
                break

            line_split = np_split(one_line.strip(), split_chars)
            #re.split(" |,", one_line.strip())
            if not initialized:
                line_length = len(line_split)
                if label_idx < 0:
                    label_idx = line_length + label_idx 

                if feat_end_idx is None:
                    if label_idx == (line_length - 1):
                        feat_end_idx = label_idx
                    else:
                        feat_end_idx = line_length

                elif feat_end_idx < 0:
                    feat_end_idx = line_length + feat_end_idx
                    
                assert feat_end_idx > (feat_start_idx + 1)
                initialized = True

            label = line_split[label_idx]
            labels.append(label)
            feats.append(np.array(line_split[feat_start_idx : feat_end_idx], dtype=np.float32))

            if len(labels) % 100000 == 0:
                print("happytools:feat_tool:read_feat_from_txt: read {} lines (splited as {})".format(len(labels), len(line_split)))

        feats = np.array(feats)
    return feats, labels

def generate_label_dict(label):
    label_sort = sorted(set(label))
    uuid_to_label = {}
    label_to_uuid = {}
    label_cum = 0
    for uuid in label_sort:
        uuid_to_label[uuid] = label_cum
        label_to_uuid[label_cum] = uuid
        label_cum += 1
    
    return uuid_to_label, label_to_uuid

def convert_feat_and_label(feat_label_list):
    concat_label = []
    for feat, label in feat_label_list:
        if feat is None:
            assert label is None
        else:
            assert label is not None
            util.EQ_CHECK(len(feat), len(label))
            concat_label.extend(label)
            
    to_new_label, to_raw_label = generate_label_dict(concat_label)
    ret = []
    for feat, label in feat_label_list:
        if feat is None:
            ret.append( (None, None) )
        else:
            ret.append( 
                ( torch.from_numpy(feat),
                    torch.from_numpy(np.array([to_new_label[i] for i in label], dtype=np.int32)) )       
            )
    
    return ret, to_raw_label

def convert_feats_to_torch(list_of_feats):
    def to_torch(feat):
        if feat is None:
            return None
        else:
            return torch.from_numpy(feat)

    ret = [
        [ to_torch(feat) for feat in feats ]
        for feats in list_of_feats
    ]
    return ret

def convert_labels_to_torch(labels):
    concat_label = []
    for label in labels:
        if label is not None:
            concat_label.extend(label)

    to_new_label, to_raw_label = generate_label_dict(concat_label)

    def to_torch(label):
        if label is None:
            return None
        else:
            return torch.from_numpy(np.array([to_new_label[i] for i in label], dtype=np.int32))

    ret = [
        to_torch(label) for label in labels
    ]
    return ret, to_raw_label


    

