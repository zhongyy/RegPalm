import numpy as np
import heapq
import pickle
import os
import torch
import time
from IPython import embed
import io
import json

# before v1.2, pytorch uses uint8 as "boolean"
torch_comparison_type = torch.bool if hasattr(torch, "bool") else torch.uint8
RAND_SEED = 1024

def TYPE_CHECK(var, var_type):
    if not type(var_type) is list:
        var_type = [var_type]

    assert type(var) in var_type, "requires {}, got {}".format(var_type, type(var))
    
def GT_CHECK(var, num, var_name="", num_name=""):
    assert var > num, "{} ({}) is not greater than {} ({})".format(var, var_name, num, num_name)

def GE_CHECK(var, num, var_name="", num_name=""):
    assert var >= num, "{} ({}) is smaller than {} ({})".format(var, var_name, num, num_name)

def LE_CHECK(var, num, var_name="", num_name=""):
    assert var <= num, "{} ({}) is greater than {} ({})".format(var, var_name, num, num_name)

def LT_CHECK(var, num, var_name="", num_name=""):
    assert var < num, "{} ({}) is not less than {} ({})".format(var, var_name, num, num_name)

def EQ_CHECK(var, num, var_name="", num_name=""):
    assert var == num, "{} ({}) is not equal to {} ({})".format(var, var_name, num, num_name)

def EXIST_CHECK(path):
    assert os.path.exists(path), "{} not exists, please manully create it".format(path)

class AutoHeap(object):
    def __init__(self, max_size, type):
        assert(max_size > 0)
        assert(type in ['MIN', 'MAX'])
        # when type is 'MIN', pop min; when 'MAX', pop max
        self.type = type
        self.max_size = max_size
        self.bank = []

    def size(self):
        return len(self.bank)
    
    def top_weight(self):
        assert(self.size())
        weight, _ = self.bank[0]
        if self.type == 'MIN':
            return weight*(1.0)
        else:
            return weight*(-1.0)
    
    def worth_push(self, weight):
        assert(type(weight) is float)
        assert(self.size())
        top_w = self.top_weight()
        if self.type == 'MIN':
            return weight > top_w
        else:
            return weight < top_w

    def push(self, weight, item):
        assert(type(weight) is float)
        if self.type == 'MIN':
            heapq.heappush(self.bank, (weight*(1.0), item))
        else:
            heapq.heappush(self.bank, (weight*(-1.0), item))
    
    def pushpop(self, weight, item):
        assert(type(weight) is float)
        if self.type == 'MIN':
            weight, item = heapq.heappushpop(self.bank, (weight*(1.0), item))
            return (weight*(1.0), item)
        else:
            weight, item = heapq.heappushpop(self.bank, (weight*(-1.0), item))
            return (weight*(-1.0), item)

    def pop(self):
        assert(self.size())
        weight, item = heapq.heappop(self.bank)
        if self.type == 'MIN':
            return (weight*(1.0), item)
        else:
            return (weight*(-1.0), item)

    def auto_push_pop(self, weight, item):
        assert(type(weight) is float)
        if self.size() < self.max_size:
            self.push(weight, item)
        elif self.worth_push(weight):
            self.pushpop(weight, item)

    def sorted_list(self):
        return list(reversed([self.pop() for _ in range(self.size())]))


def non_zero_start_end(arr):
    result_start = 0
    result_end = len(arr)
    for i in range(result_start, len(arr)):
        if arr[i] != 0:
            result_start = i
            break  
    
    for i in range(result_start, len(arr)):
        if arr[i] == 0:
            result_end = i
            break  
    
    return (result_start, result_end)

class RocContainer(object):
    def __init__(self, vectors, meta=None, buckets=None, min_val=-1.0, max_val=1.0):
        TYPE_CHECK(vectors, dict)
        self.vectors = vectors
        for i in self.vectors:
            TYPE_CHECK(self.vectors[i], np.ndarray)
            EQ_CHECK(self.vectors[i].dtype, np.float32)
            EQ_CHECK(self.vectors[i].ndim, 1)

        for key in ["tp", "fp"]:
            assert key in self.vectors

        self.tpr = self.vectors["tp"]
        self.fpr = self.vectors["fp"]
        self.outlier = self.vectors["out"] if "out" in self.vectors else None
        self.bins = len(self.tpr)
        for i in self.vectors:
            EQ_CHECK(len(self.vectors[i]), self.bins)

        self.meta = meta
        self.mode = "1vs1" if self.outlier is None else "1vsN"
        self.buckets = buckets
        self.min_val = min_val
        self.max_val = max_val

    def get_N(self):
        # return gallery size
        if self.mode is "1vsN":
            return self.meta[0]
        else:
            raise ValueError("{} mode has no information about N".format(self.mode))
    
    def tpr_and_fpr(self, outlier_ratio=None):
        if self.mode is "1vs1":
            non_zero_st, non_zero_ed = non_zero_start_end(self.fpr)
            return (self.tpr[non_zero_st: non_zero_ed], self.fpr[non_zero_st: non_zero_ed]) 
        else:
            assert outlier_ratio is not None, "please set a outlier_ratio"
            outlier_ratio = float(outlier_ratio)
            fpr_fusion = (self.fpr  + self.outlier * (outlier_ratio)) / (1 + outlier_ratio)
            non_zero_st, non_zero_ed = non_zero_start_end(fpr_fusion)
            return (self.tpr[non_zero_st: non_zero_ed], fpr_fusion[non_zero_st: non_zero_ed]) 

    def data(self):
        if self.mode is "1vsN":
            return (self.tpr, self.fpr, self.outlier) 
        else:
            return (self.tpr, self.fpr) 

    def thres_to_index(self, thres):
        thres = float(thres)
        return int((thres - self.min_val) / (self.max_val - self.min_val) * self.bins)
    
    def index_to_thres(self, index):
        index = float(index)
        return (index / self.bins) * (self.max_val - self.min_val) + self.min_val
    
    def index_to_score(self, index):
        index = float(index)
        return (index / self.bins) * 100.0

    def score_to_thres(self, score):
        return score / 100.0 * (self.max_val - self.min_val) + self.min_val

    def tpr_at_thres(self, thres):
        assert(thres >= -1 and thres < 1)
        return self.tpr[self.thres_to_index(thres)]

    def tpr_at_score(self, score):
        assert(score >= 0 and score < 100)
        thres = self.score_to_thres(score)
        return self.tpr_at_thres(thres)

    def fpr_at_thres(self, thres, outlier_ratio=None):
        assert(thres >= -1 and thres < 1)
        if self.mode is "1vs1":
            return self.fpr[self.thres_to_index(thres)]
        else:
            assert outlier_ratio is not None
            outlier_ratio = float(outlier_ratio)
            fpr_fusion = (self.fpr  + self.outlier * (outlier_ratio)) / (1 + outlier_ratio)
            return fpr_fusion[self.thres_to_index(thres)]

    def fpr_at_score(self, score, outlier_ratio=None):
        assert(score >= 0 and score < 100)
        thres = self.score_to_thres(score)
        return self.fpr_at_thres(thres, outlier_ratio)

    def tpr_at_fpr(self, fpr, outlier_ratio=None):
        if self.mode is "1vs1":
            fpr_fusion = self.fpr
        else:
            assert outlier_ratio is not None
            outlier_ratio = float(outlier_ratio)
            fpr_fusion = (self.fpr  + self.outlier * (outlier_ratio)) / (1 + outlier_ratio)
        
        for i in range(len(fpr_fusion)):
            if fpr_fusion[i] < fpr:
                return (self.tpr[i], fpr_fusion[i], self.index_to_score(i))
    
    def fpr_at_tpr(self, tpr, outlier_ratio=None):
        if self.mode is "1vs1":
            fpr_fusion = self.fpr
        else:
            assert outlier_ratio is not None
            outlier_ratio = float(outlier_ratio)
            fpr_fusion = (self.fpr  + self.outlier * (outlier_ratio)) / (1 + outlier_ratio)
        
        for i in range(len(self.tpr)):
            if self.tpr[i] < tpr:
                return (self.tpr[i], fpr_fusion[i], self.index_to_score(i))

    def dump(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

        for key in self.vectors:
            self.vectors[key].tofile(os.path.join(folder, "{}_vector.bin".format(key)))

        if self.meta is not None:
            self.meta.tofile(os.path.join(folder, 'meta.bin'))

        if self.buckets is not None:
            buckets_as_dict = { k : self.buckets[k].data for k in self.buckets }
            with open(os.path.join(folder, 'buckets.pkl'), "wb") as f:
                pickle.dump(buckets_as_dict, f, protocol=2)

        min_max_js = json.dumps({"min_val": self.min_val, "max_val": self.max_val})
        with open(os.path.join(folder, 'min_max_js.txt'), "w") as f:
            f.write("{}\n".format(min_max_js))

    """
    def cos_at_fpr(self, fpr):
        _, _, i = self.tpr_at_fpr(fpr)
        return float(i) / self.bins * 2 - 1
    
    def cos_at_tpr(self, tpr):
        _, _, i = self.fpr_at_tpr(tpr)
        return float(i) / self.bins * 2 - 1
    """

class Bucket(object):
    r"""
    a list, with a limited volume
    """
    def __init__(self, volume=None, data=[]):
        self.volume = volume if volume is not None else len(data)
        self.data = data[ : self.volume]

    def add(self, inp):
        for i in inp:
            if len(self.data) >= self.volume:
                break
            else:
                self.data.append(i)

    def get_data(self):
        return self.data

    def merge(self, backets):
        TYPE_CHECK(backets, list)
        GE_CHECK(len(backets), 1)
        TYPE_CHECK(backets[0], Bucket)
        for backet in backets:
            self.data.extend(backet.get_data())
        
        np.random.shuffle(self.data)
        self.data = self.data[ : self.volume]
        return self.data


def merge_rocs(roc_list):
    r""" Merge multiple rocs to one, by calculate the average of 
    their vectors (including tp, fp), and collect appendixes all
    together
    """
    GE_CHECK(len(roc_list), 1)
    vector_keys = set(roc_list[0].vectors.keys())
    for roc in roc_list:
        EQ_CHECK(set(roc.vectors.keys()), vector_keys)
          
    merged_vectors = {
        key : sum([roc.vectors[key] for roc in roc_list]) / len(roc_list)
        for key in vector_keys
    }

    if roc_list[0].buckets is not None:
        bucket_keys = set(roc_list[0].buckets.keys())
        for roc in roc_list:
            EQ_CHECK(set(roc.buckets.keys()), bucket_keys)

        merged_buckets = {
            key : Bucket(volume=roc_list[0].buckets[key].volume)
            for key in bucket_keys
        }

        for key in bucket_keys:
            merged_buckets[key].merge([roc.buckets[key] for roc in roc_list])

    else:
        for roc in roc_list:
            assert roc.buckets is None

        merged_buckets = None

    merged_meta = sum([roc.meta for roc in roc_list]) // len(roc_list)

    return RocContainer(vectors=merged_vectors, meta=merged_meta, buckets=merged_buckets)

def generate_gallery(N_sets, num_template, repeats=5):
    TYPE_CHECK(N_sets, list)
    GT_CHECK(len(N_sets), 0)
    LE_CHECK(len(N_sets), 11, "length of N sets", "max supported length")
    N_sets = sorted(set(N_sets))
    if type(repeats) is list:
        EQ_CHECK(len(N_sets), len(repeats), "length of N sets", "length of sample times")
    else:
        GT_CHECK(repeats, 0)
        repeats = [repeats for _ in range(len(N_sets))]

    galleries = []
    N_and_times = zip(N_sets, repeats)
    for N, times in N_and_times:
        TYPE_CHECK(N, int)
        GE_CHECK(N, 10, "gallery size", "minimum gallery size")
        LE_CHECK(N, num_template, "gallery size", "total gallery size")
        TYPE_CHECK(times, int)
        GT_CHECK(times, 0)
        seeds = range(RAND_SEED, RAND_SEED + times)
        for seed in seeds:
            np.random.seed(seed)
            tmp_arange = np.arange(num_template)
            np.random.shuffle(tmp_arange)
            galleries.append( { "index": tmp_arange[:N] } )
            
    assert len(galleries) > 0, "galleries is empty!!!"
    return galleries

def check_roc_input(feat_label_list):
    feat_len = None
    for feat, label, caption, required, shape_check in  feat_label_list:  
        if required:
            assert feat is not None
            assert label is not None

        if (feat is not None) and (label is not None):
            TYPE_CHECK(feat, np.ndarray)
            TYPE_CHECK(label, list)
            EQ_CHECK(feat.dtype, np.float32)
            EQ_CHECK(feat.shape[0], len(label), "feature count", "label count")

            if shape_check:
                EQ_CHECK(feat.ndim, 2) 
                if feat_len is None:
                    feat_len = feat.shape[1]
                else:
                    EQ_CHECK(feat.shape[1], feat_len, "feat length for {}".format(caption), "existing feat length")

            print("{} checked".format(caption))
        else:
            print("{} is absence".format(caption))

def check_visual(visual_config):
    assert "fp_thres" in visual_config, "please give a fp threshold, between (-1, 1)"
    assert "tp_thres" in visual_config, "please give a tp threshold, between (-1, 1)"
    assert "volume" in visual_config, "please give a volume (capacity) [1, 1000000]"
    LE_CHECK(visual_config["fp_thres"], 1.0)
    LE_CHECK(visual_config["tp_thres"], 1.0)
    GE_CHECK(visual_config["fp_thres"], -1.0)
    GE_CHECK(visual_config["tp_thres"], -1.0)
    LE_CHECK(visual_config["volume"], 1000000)
    GE_CHECK(visual_config["volume"], 1)

class ListGenerator(object):
    def __init__(self, data):
        TYPE_CHECK(data, list)
        self.data = data
        self.length = len(data)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self._gen()

    def _gen(self):
        for i in self.data:
            yield i


class Tiktok(object):
    def __init__(self):
        self.t = time.time()

    def tik(self):      
        self.t = time.time()

    def tok(self, caption=""):   
        old_t = self.t
        self.t = time.time()
        print("Tiktok INFO - {}: {}".format(caption, self.t - old_t))

def _readlines(filename):
    with open(filename, "r") as f:
        return f.readlines()

def load_roc(folder):    
    # compatible with old-version
    if os.path.exists(os.path.join(folder, 'tpr.bin')):
        tpr = np.fromfile(os.path.join(folder, 'tpr.bin'), 
            dtype=np.float32).reshape(-1)
        fpr = np.fromfile(os.path.join(folder, 'fpr.bin'), 
            dtype=np.float32).reshape(-1)
        vectors = {"tp" : tpr, "fp" : fpr}
        outlier_path = os.path.join(folder, 'out.bin')
        if os.path.exists(outlier_path):
            outlier = np.fromfile(outlier_path, dtype=np.float32).reshape(-1)
            vectors["out"] = outlier
    else:
        vector_paths = [i for i in os.listdir(folder) if i[-11:] == "_vector.bin"]
        vectors = {
            i[:-11]: np.fromfile(os.path.join(folder, i), dtype=np.float32).reshape(-1)
            for i in vector_paths
        }

    meta_path = os.path.join(folder, 'meta.bin')
    meta = np.fromfile(meta_path, dtype=np.int32).reshape(-1) \
        if os.path.exists(meta_path) else None

    buckets = None
    buckets_path = os.path.join(folder, 'buckets.pkl')
    if os.path.exists(buckets_path):
        with open(buckets_path, "rb") as f:
            buckets_dict = pickle.load(f)
        
        buckets = {k : Bucket(data=buckets_dict[k]) for k in buckets_dict}

    max_min_dict = {
        "min_val" : -1.0,
        "max_val" : 1.0,
    } if not os.path.exists(
        os.path.join(folder, 'min_max_js.txt')
    ) else json.loads(
        _readlines(os.path.join(folder, 'min_max_js.txt'))[0].strip()
    )

    return RocContainer(
        vectors=vectors, meta=meta, buckets=buckets, 
        min_val=max_min_dict["min_val"], max_val=max_min_dict["max_val"]
    )

def load_roc_from_txt(txt):
    with io.open(txt, 'r') as f:
        lines = f.readlines()
        lines = [tuple(i.strip().split()) for i in lines]

    tpr = np.array([i[0].strip(',') for i in lines], dtype=np.float32)
    fpr = np.array([i[1].strip(',') for i in lines], dtype=np.float32)
    
    return RocContainer({"tp" : tpr, "fp" : fpr})


