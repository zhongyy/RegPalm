import torch
from . import util
from inspect import isfunction
import numpy as np
from IPython import embed
import math

def cosine_similarity(a, b):
    r""" This function calculates cosine similarity.

    Different to torch.nn.functional.cosine_similarity, this function every item in  
    Cartesian product of a and b, i.e., a matrix of size len(a) * len(b) is returned.
    
    Arguments:
        a (torch.Tensor, required): feature matrix (len(a) * N), where N is feat length
        b (torch.Tensor, required): feature matrix (len(b) * M), where M should equal N
        out: Tensor of size len(a) * len(b)
    """
    assert type(a) is torch.Tensor, "a type error, {} vs torch.Tensor".format(type(a))
    assert type(b) is torch.Tensor, "b type error, {} vs torch.Tensor".format(type(b))
    assert a.dim() == 2, "a is not a 2-D mat, dim = {}".format(a.dim())
    assert b.dim() == 2, "b is not a 2-D mat, dim = {}".format(b.dim())
    assert a.dtype == torch.float32, "a is not a float32 tensor, {}".format(a.dtype)
    assert b.dtype == torch.float32, "b is not a float32 tensor, {}".format(b.dtype)
    assert a.size(-1) == b.size(-1), "different feature dim, {} vs {}".format(a.size(-1), b.size(-1))
    matmul = torch.matmul(a, b.t())
    a_norm_sqrt = torch.sqrt(torch.sum(torch.mul(a, a), 1, keepdim=True))
    b_norm_sqrt = torch.sqrt(torch.sum(torch.mul(b, b), 1, keepdim=True))
    norm_sqrt_mul = torch.matmul(a_norm_sqrt, b_norm_sqrt.t())
    return torch.clamp( torch.div(matmul, norm_sqrt_mul), -0.99999, 0.99999)

def pseudo_int8_cosine_similarity(a, b):
    r""" This function calculates cosine similarity.

    Different to torch.nn.functional.cosine_similarity, this function every item in  
    Cartesian product of a and b, i.e., a matrix of size len(a) * len(b) is returned.
    
    Arguments:
        a (torch.Tensor, required): feature matrix (len(a) * N), where N is feat length
        b (torch.Tensor, required): feature matrix (len(b) * M), where M should equal N
        out: Tensor of size len(a) * len(b)
    """
    assert type(a) is torch.Tensor, "a type error, {} vs torch.Tensor".format(type(a))
    assert type(b) is torch.Tensor, "b type error, {} vs torch.Tensor".format(type(b))
    assert a.dim() == 2, "a is not a 2-D mat, dim = {}".format(a.dim())
    assert b.dim() == 2, "b is not a 2-D mat, dim = {}".format(b.dim())
    assert a.dtype == torch.float32, "a is not a float32 tensor, {}".format(a.dtype)
    assert b.dtype == torch.float32, "b is not a float32 tensor, {}".format(b.dtype)
    assert a.size(-1) == b.size(-1), "different feature dim, {} vs {}".format(a.size(-1), b.size(-1))
    
    a_norm_sqrt = torch.sqrt(torch.sum(torch.mul(a, a), 1, keepdim=True))
    b_norm_sqrt = torch.sqrt(torch.sum(torch.mul(b, b), 1, keepdim=True))
    a = torch.div(a, a_norm_sqrt)
    b = torch.div(b, b_norm_sqrt)
    norm = 512.0
    a = torch.clamp(a * norm, -127.0, 127.0).int().float()
    b = torch.clamp(b * norm, -127.0, 127.0).int().float()
    matmul = torch.matmul(a, b.t())
    return torch.clamp( matmul / (norm ** 2), -0.99999, 0.99999)

def int8_cosine_similarity(a, b):
    r""" This function calculates cosine similarity.

    Different to torch.nn.functional.cosine_similarity, this function every item in  
    Cartesian product of a and b, i.e., a matrix of size len(a) * len(b) is returned.
    
    Arguments:
        a (torch.Tensor, required): feature matrix (len(a) * N), where N is feat length
        b (torch.Tensor, required): feature matrix (len(b) * M), where M should equal N
        out: Tensor of size len(a) * len(b)
    """
    assert type(a) is torch.Tensor, "a type error, {} vs torch.Tensor".format(type(a))
    assert type(b) is torch.Tensor, "b type error, {} vs torch.Tensor".format(type(b))
    assert a.dim() == 2, "a is not a 2-D mat, dim = {}".format(a.dim())
    assert b.dim() == 2, "b is not a 2-D mat, dim = {}".format(b.dim())
    assert a.dtype == torch.float32, "a is not a float32 tensor, {}".format(a.dtype)
    assert b.dtype == torch.float32, "b is not a float32 tensor, {}".format(b.dtype)
    assert a.size(-1) == b.size(-1), "different feature dim, {} vs {}".format(a.size(-1), b.size(-1))
    
    norm = 512.0
    matmul = torch.matmul(a, b.t())
    cosine = torch.clamp( matmul / (norm ** 2), -0.99999, 0.99999)
    return cosine

def specific_cosine_similarity(a, b, func):
    if func is "float":
        return cosine_similarity(a, b)
    elif func is "pseudo_int8":
        return pseudo_int8_cosine_similarity(a, b)
    elif func is "int8":
        return int8_cosine_similarity(a, b)
    else:
        raise ValueError("{} cosine-mode not support".format(func))


def get_indicate_matrix(a, b, OP="EQ"):
    r""" This function returns a positive matrix between a and b, 
    with 1 representing positive(equal at that position) and 0 being negative(unequal).                                                               

    Arguments:
        a (torch.Tensor, required): 1-D matrix, dtype=int32
        b (torch.Tensor, required): 1-D matrix, dtype=int32
        out: Tensor of size len(a) * len(b) 
    """
    assert type(a) is torch.Tensor, "a type error, {} vs torch.Tensor".format(type(a))
    assert type(b) is torch.Tensor, "b type error, {} vs torch.Tensor".format(type(b))
    assert a.dim() == 1, "a is not a 1-D mat, dim = {}".format(a.dim())
    assert b.dim() == 1, "b is not a 1-D mat, dim = {}".format(b.dim())

    if OP in ["EQ", "GE", "GT", "LE", "LT"]:
        assert a.dtype in [torch.int32, torch.int64, torch.int16, torch.int8, util.torch_comparison_type]
    elif OP in ["AND"]:
        assert a.dtype == util.torch_comparison_type
    
    util.EQ_CHECK(a.dtype, b.dtype)
    if OP == "EQ":
        return torch.eq(a.view(-1, 1), b.view(1, -1))
    elif OP == "GE":
        return torch.ge(a.view(-1, 1), b.view(1, -1))
    elif OP == "GT":
        return torch.gt(a.view(-1, 1), b.view(1, -1))
    elif OP == "LE":
        return torch.le(a.view(-1, 1), b.view(1, -1))    
    elif OP == "LT":
        return torch.lt(a.view(-1, 1), b.view(1, -1))  
    elif OP == "AND":
        return torch.mul(a.view(-1, 1), b.view(1, -1))
    else:
        assert False, "Operator {} is not defined".format(OP)  


def get_patches(m, n, step=2000, down_triangle=False):
    r""" This function returns a list of patches, which are sliced from a big matrix.
    
    Arguments:
        m : (int, required) num of matrix-row(axis-0)
        n : (int, required) num of matrix-col(axis-1)
        step : (int, optional) step size of a patch
        out: list, each is a tuple ((row_start, row_end),(col_start, col_end)),
            start is included while end is excluded.
    """
    assert type(m) is int, "m type error, {} vs int".format(type(m))
    assert type(m) is int, "n type error, {} vs int".format(type(n))
    assert type(step) is int, "step type error, {} vs int".format(type(step))
    assert step > 0, "step should be greater than 0, {}".format(step)
    assert m > 0 and n > 0, "M and N should be greater than 0, {} {}".format(m, n)
    if down_triangle:
        assert m == n, "down_triangle requires m == n"

    start_points_along_m = range(0, m, step)
    start_points_along_n = range(0, n, step)
    # for example range(0,100,50) returns [0,50], range(0,101,50) returns [0,50,100]

    # generate patches.
    # normally, each rectangle-patch's size is (step, step).
    # in down_triangle mode, a patch is thought to be valid, 
    # only if its bottom-left conner is below (excluded) the diagonal line.
    patches = []
    for i in start_points_along_m:
        for j in start_points_along_n:
            m_patch = (i, min(m, i+step))
            n_patch = (j, min(n, j+step))
            if down_triangle:
                bottom_left_row = m_patch[1] - 1
                bottom_left_col = n_patch[0]
                if bottom_left_row > bottom_left_col:
                    patches.append((m_patch, n_patch))
            else:
                patches.append((m_patch, n_patch))

    return patches

def mask_histc(src, mask, bins, min_val, max_val):
    util.EQ_CHECK(mask.dtype, util.torch_comparison_type)
    util.EQ_CHECK(mask.ndimension(), 1)
    util.EQ_CHECK(src.ndimension(), 1)
    util.EQ_CHECK(src.size(), mask.size())
    src_masked = src[mask.nonzero().unbind(dim=1)[0]]
    # "torch.histc" has only cpu-version for early pytorch
    src_masked = src_masked.cpu()
    return torch.histc(src_masked, bins=bins, min=min_val, max=max_val).to(torch.float64)
    
def simple_mean(scores):
    return sum(scores) / float(len(scores))

native_similarity_funcs = {
    "float" : cosine_similarity, 
    "pseudo_int8" : pseudo_int8_cosine_similarity,
    "int8" : int8_cosine_similarity,
}

def compatible_simfun(similarity_func):
    if type(similarity_func) is str:
        assert similarity_func in native_similarity_funcs, "{} is not suppoted".format(similarity_func)
        similarity_func = native_similarity_funcs[similarity_func]
    else:
        assert isfunction(similarity_func), "need function type"
    
    return similarity_func


class Top3FetcherBase(object):
    def __init__(self):
        return
    
    def fetch(self, sample_feats, template_feats, sample_flags, template_flags):
        raise NotImplementedError

class Top3FetcherByScoreFusion(Top3FetcherBase):
    def __init__(self, weights=None):
        super(Top3FetcherByScoreFusion, self).__init__()
        print("Init top3-fetcher with score fusion mode")
        if weights is not None:
            assert type(weights) in (list, tuple), "weights type error, {} vs (list or tuple)".format(type(weights))
            assert sum(weights) == 1, "sum of weights should be excactly 1, got {}".format(sum(weights))

        self.weights = weights
    
    def fetch(self, 
        sample_feats, template_feats, 
        sample_flags, template_flags, 
        similarity_funcs):

        flag_mats = [
            get_indicate_matrix(s_flag, t_flag, OP="AND")
            for s_flag, t_flag in zip(sample_flags, template_flags)
        ]

        raw_scores = [
            sim_fun(s_feat, t_feat)
            for s_feat, t_feat, sim_fun in zip(
                sample_feats, 
                template_feats, 
                similarity_funcs)
        ] 
        if self.weights is None:
            self.weights = tuple(list([1.0 / len(raw_scores) for _ in range(len(raw_scores))]))

        
        assert len(self.weights) == len(raw_scores), \
            "len of weights ({}) not match len of modalities {}".format(
                len(self.weights), len(raw_scores)
            )

        fused_score = sum([i * j.float() * k for i, j, k in zip(raw_scores, flag_mats, self.weights)]) / \
            (sum([i.float() * j for i, j in zip(flag_mats, self.weights)]) + 0.0001)
        
        assert fused_score.max().item() < 1.0 and fused_score.min().item() > -1.0, "fatal error"
        top3_score, top3_ind = fused_score.topk(k=3)
        return top3_score, top3_ind


class FpTable(object):
    MAX_TABLE_LENGTH = 30
    def __init__(self, fp_table):
        util.TYPE_CHECK(fp_table, list)
        util.GE_CHECK(len(fp_table), 10)
        util.EQ_CHECK(len(fp_table) % 2, 0)
        table = [ 
            ( float(fp_table[i]), float(fp_table[i+1]) ) 
            for i in range(0, len(fp_table), 2) 
        ]
        for line in table:
            util.GT_CHECK(line[0], 0.0)
            util.LT_CHECK(line[0], 1.0)
            util.GT_CHECK(line[1], 0.001)
            util.LT_CHECK(line[1], 99.999)
        
        # sort by score
        table = sorted(table, key=lambda x: x[1])

        valid_table = []
        for item in table:
            fp, _ = item
            if len(valid_table) == 0:
                valid_table.append(item)
            elif math.fabs(fp - valid_table[-1][0]) / valid_table[-1][0] > 0.1:
                valid_table.append(item)
        
        table = valid_table

        sample_num = min(self.MAX_TABLE_LENGTH, len(table))
        sample_index = np.linspace(0, len(table)-1, sample_num).astype(np.int64)
        table = [
            table[i] 
            for i in sample_index.tolist()
        ]

        """
        Now, the table looks like 
        [
            (1e-3, 60),
            (1e-4, 70),
            (1e-5, 80),
        ]
        """
        lower_fp = [i[0] for i in table[1: ]]
        upper_fp = [i[0] for i in table[:-1]]
        lower_score = [i[1] for i in table[:-1]]
        upper_score = [i[1] for i in table[1: ]] 
        table = list(zip(lower_fp, upper_fp, lower_score, upper_score))
        """
        [
            (1e-4, 1e-3, 60,    70     ),
            (1e-5, 1e-4, 70,    80     ),
        ]
        """
        self.table = table
    
    def get_fp_lowbound(self):
        return self.table[-1][0]

    def get_fp_upbound(self):
        return self.table[0][1]
    
    def get_score_lowbound(self):
        return self.table[0][2]

    def get_score_upbound(self):
        return self.table[-1][3]

    def score_to_fp(self, score_tensor):
        slices = [
            (torch.gt(score_tensor, lower_score) & torch.le(score_tensor, upper_score)).float() * upper_fp
            for _, upper_fp, lower_score, upper_score in self.table
        ]
        slices.extend([
            torch.gt(score_tensor, self.get_score_upbound()).float() * self.get_fp_lowbound(),
            torch.le(score_tensor, self.get_score_lowbound()).float() * 0.9999,
        ])

        fp = sum(slices)
        return fp

    def fp_to_score(self, fp_tensor):
        slices = [
            (torch.gt(fp_tensor, lower_fp) & torch.le(fp_tensor, upper_fp)).float() * \
                ((upper_score - lower_score) * (upper_fp - fp_tensor) / (upper_fp - lower_fp) + lower_score)
            for lower_fp, upper_fp, lower_score, upper_score in self.table
        ]
        slices.extend([
            torch.gt(fp_tensor, self.get_fp_upbound()).float() * 0.001,
            torch.le(fp_tensor, self.get_fp_lowbound()).float() * self.get_score_upbound(),
        ])

        score = sum(slices)
        return score

class Top3FetcherByFpFusion(Top3FetcherBase):
    def __init__(self, fp_tables, coefficients, rough_sort_k=10):
        super(Top3FetcherByFpFusion, self).__init__()
        print("Init top3-fetcher with FP(false positive rate) fusion mode")
        util.TYPE_CHECK(rough_sort_k, int)
        util.GE_CHECK(rough_sort_k, 1)
        util.TYPE_CHECK(fp_tables, list)
        util.TYPE_CHECK(coefficients, list)
        self.rough_sort_k = rough_sort_k
        self.fp_tables = [FpTable(table) for table in fp_tables]
        self.n_modal = len(self.fp_tables)
        util.EQ_CHECK(len(coefficients), self.n_modal - 1)
        self.coefficients = coefficients
        self.coefficients.insert(0, 1)

    def fetch(self, 
        sample_feats, template_feats, 
        sample_flags, template_flags, 
        similarity_funcs):

        util.EQ_CHECK(len(sample_feats), self.n_modal)
        util.EQ_CHECK(len(template_feats), self.n_modal)
        util.EQ_CHECK(len(sample_flags), self.n_modal)
        util.EQ_CHECK(len(template_flags), self.n_modal)

        # 0th modal cannot be absence
        util.EQ_CHECK(sample_flags[0].min().cpu().int().item(), 1)
        util.EQ_CHECK(template_flags[0].min().cpu().int().item(), 1)

        raw_scores = [
            sim_fun(s_feat, t_feat)
            for s_feat, t_feat, sim_fun in zip(
                sample_feats, 
                template_feats, 
                similarity_funcs)
        ]
        for score in raw_scores:
            assert score.max().item() < 1.0 and score.min().item() > -1.0, "fatal error, ask @junan"
        
        flag_mats = [
            get_indicate_matrix(s_flag, t_flag, OP="AND")
            for s_flag, t_flag in zip(sample_flags, template_flags)
        ]
        """
        raw_scores = [ 
            torch.tensor([
                [10.0, 0.0, 99.9, 75, 72],
                [65, 75.0, 80, 72, 85],
            ]) / 50 - 1,
            torch.tensor([
                [75, 85.0, -40, 75, 85],
                [75, 85.0, 70, 75, 80],
            ]) / 50 - 1,
        ]

        flag_mats = [
            torch.tensor([
                [True, True, True, True, True],
                [True, True, False, True, True],
            ]),
            torch.tensor([
                [True, True, True, True, True],
                [True, True, True, False, True],
            ]),
        ]
        """
        # perform rough sorting, by the first modality
        _, rough_topk_ind = raw_scores[0].topk(k=self.rough_sort_k)

        #print(rough_topk_ind)

        rough_topk_scores = [torch.gather(score, 1, rough_topk_ind) for score in raw_scores]
        rough_topk_flags = [torch.gather(flag, 1, rough_topk_ind) for flag in flag_mats]

        #print(rough_topk_scores)
        #print(rough_topk_flags)

        fine_topk_score = self.fp_fusion(rough_topk_scores, rough_topk_flags)

        #print(fine_topk_score)

        top3_score, top3_ind = fine_topk_score.topk(k=3)
        return top3_score, torch.gather(rough_topk_ind, 1, top3_ind)
    
    def fp_fusion(self, score_matrixes, flag_matrixes):
        fps = [
            (fp_table.score_to_fp(score * 50.0 + 50.0) * flag.float() +  (1.0 / coef) * (~flag).float()) * coef
            for fp_table, score, flag, coef in zip(
                self.fp_tables, 
                score_matrixes, 
                flag_matrixes,
                self.coefficients
                )
        ]

        # product them all 
        fused_fp = torch.prod(
            torch.cat([i.view(1, -1) for i in fps]), 
            dim=0
        ).view(fps[0].size())

        fused_score = self.fp_tables[0].fp_to_score(fused_fp)
        return (fused_score - 50.0) / 50.0
