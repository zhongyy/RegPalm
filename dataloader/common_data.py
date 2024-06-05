import numpy as np
import scipy.misc
import os
import torch
from PIL import Image
from io import BytesIO
import base64
import torchvision

from .diy_transforms import RandomResizedPaddingCrop
class _AugmentorTransform(torch.nn.Module):
    """Wrap on Augmentor Pipeline
    """

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: .
        """

        for operation in self.pipeline.operations:
            if torch.rand(1) < operation.probability:
                img = operation.perform_operation([img])[0]

        return img

    def __repr__(self):
        return self.__class__.__name__

def build_train_test_transform(
        aug_input_size=242, input_size=224, scale=(0.78, 0.86), gray_scale=False, random_skew=False,
        overcrop_input_size_w=None, overcrop_input_size_h=None
    ):
    print('Building train transforms ...')
    """
        Original input_size(224), aug_input_size(242) is set for w270xh260 input
        because 260 / 240 * 224 == 242, 240 * 240 / (260 * 270) == 0.82
    """
    
    #norm_mean=[0.485, 0.456, 0.406]
    #norm_std=[0.229, 0.224, 0.225]
    norm_mean=[0.0, 0.0, 0.0]
    norm_std=[1.0, 1.0, 1.0]
    normalize = torchvision.transforms.Normalize(mean=norm_mean, std=norm_std)
    gray_normalize = torchvision.transforms.Normalize(mean=[0.0], std=[1.0])

    transform_tr = []
    transform_te = []

    transform_tr += [torchvision.transforms.Resize(aug_input_size)] # keeping ratio, short-edge to input_size
    if random_skew:
        print("use random_skew")
        import Augmentor
        aug_pipe = Augmentor.Pipeline()
        #aug_pipe.skew(0.5, 0.1)
        aug_pipe.rotate(0.5, 5, 5)
        transform_tr += [_AugmentorTransform(pipeline=aug_pipe)]

    if (overcrop_input_size_w is not None) or (overcrop_input_size_h is not None):
        assert (overcrop_input_size_w is not None) and (overcrop_input_size_h is not None)
        assert overcrop_input_size_w >= input_size
        assert overcrop_input_size_h >= input_size
        transform_tr += [RandomResizedPaddingCrop(
            input_size, 
            overcrop_input_h=overcrop_input_size_h, 
            overcrop_input_w=overcrop_input_size_w,
            scale=scale, ratio=(0.8333, 1.2)
        )]

        transform_te += [torchvision.transforms.Resize(aug_input_size)] # keeping ratio, short-edge to input_size
        transform_te += [torchvision.transforms.CenterCrop((overcrop_input_size_h, overcrop_input_size_w))]
    else:
        transform_tr += [torchvision.transforms.RandomResizedCrop((input_size, input_size), scale=scale, ratio=(0.8333, 1.2))]
    
        transform_te += [torchvision.transforms.Resize(aug_input_size)] # keeping ratio, short-edge to input_size
        transform_te += [torchvision.transforms.CenterCrop(input_size)]

    transform_tr += [
        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.075)
    ]
    transform_tr += [torchvision.transforms.RandomGrayscale()]

    if gray_scale:
        transform_tr += [torchvision.transforms.Grayscale(num_output_channels=1)]
        transform_te += [torchvision.transforms.Grayscale(num_output_channels=1)]

    transform_tr += [torchvision.transforms.ToTensor()]
    transform_te += [torchvision.transforms.ToTensor()]

    if gray_scale:
        transform_tr += [gray_normalize]
        transform_te += [gray_normalize]
    else:
        transform_tr += [normalize]
        transform_te += [normalize]

    transform_tr = torchvision.transforms.Compose(transform_tr)
    transform_te = torchvision.transforms.Compose(transform_te)

    print(transform_te)
    return transform_tr, transform_te


import timm
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import _select_rand_weights, rand_augment_ops, RandAugment, AugmentOp

_RAND_TRANSFORMS = [
    'AutoContrast',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Sharpness',
]

_RAND_INCREASING_TRANSFORMS = [
    'AutoContrast',
    'SolarizeAdd',
    'ColorIncreasing',
    'ContrastIncreasing',
    'SharpnessIncreasing',
]


def AugmentOpStr(self):
    # return "{}(prob={},magnitude={},hparams={})".format(self.name if hasattr(self, 'name') else self.aug_fn, self.prob, self.magnitude, self.hparams)
    return "{}(prob={},magnitude={},hparams={})".format(self.aug_fn.__name__, self.prob, self.magnitude, self.hparams)

def RandAugmentStr(self):
    return 'RandAugment(num_layers={}, choice_weights={}, ops=\n    {})'.format(self.num_layers, self.choice_weights, '\n    '.join([str(op) for op in self.ops]))

def RandomErasingStr(self):
    return 'RandomErasing(prob={}, max_area={:.2f}, max_count={})'.format(self.probability, self.max_area, self.max_count)

timm.data.auto_augment.AugmentOp.__str__ = AugmentOpStr
timm.data.auto_augment.RandAugment.__str__ = RandAugmentStr
timm.data.random_erasing.RandomErasing.__str__ = RandomErasingStr

def build_geo_transform(input_size, aa_geo, geo_transforms, mean):
    import re
    def rand_augment_transform(config_str, hparams):
        """
        Create a RandAugment transform

        :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
        dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
        sections, not order sepecific determine
            'm' - integer magnitude of rand augment
            'n' - integer num layers (number of transform ops selected per image)
            'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
            'mstd' -  float std deviation of magnitude noise applied
            'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
        Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
        'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2

        :param hparams: Other hparams (kwargs) for the RandAugmentation scheme

        :return: A PyTorch compatible Transform
        """
        magnitude = 9 # default to _MAX_LEVEL for magnitude (currently 10)
        num_layers = 2  # default to 2 ops per image
        prob = 0.5
        weight_idx = None  # default to no probability weights for op choice

        config = config_str.split('-')
        assert config[0] == 'rand'
        config = config[1:]
        for c in config:
            cs = re.split(r'(\d.*)', c)
            if len(cs) < 2:
                continue
            key, val = cs[:2]
            if key == 'mstd':
                # noise param injected via hparams for now
                hparams.setdefault('magnitude_std', float(val))
            elif key == 'inc':
                pass
            elif key == 'm':
                magnitude = int(val)
            elif key == 'n':
                num_layers = int(val)
            elif key == 'w':
                weight_idx = int(val)
            elif key == 'p':
                prob = float(val)
            else:
                assert False, 'Unknown RandAugment config section'
        # ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=geo_transforms)
        ra_ops = [AugmentOp(name, prob=prob, magnitude=magnitude, hparams=hparams) for name in geo_transforms]

        choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
        return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)

    geo_params = dict(
        translate_const=int(input_size * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in mean]),
    )
    geo_transform = rand_augment_transform(aa_geo, geo_params)

    return geo_transform

def build_train_test_lux_transform(
    aug_input_size=242, input_size=224, scale=(0.78, 0.86), overcrop_input_size_w=None, overcrop_input_size_h=None
):
    print('Building lux transforms ...')
    mean = [0,0,0]
    std = [1,1,1]

    param_aug_input_size = aug_input_size
    param_input_size = input_size
    param_crop_scale = scale
    param_crop_aspect_ratio = (0.8333, 1.2)
    param_reprob = 0.01
    param_rearea = 0.01
    param_aa = "rand-m5-mstd0.5-inc1"
    param_aa_geo = "rand-m2-mstd0.5-inc1-p0.25" 
    param_aa_geo_trans = "rand-m1-mstd0.5-inc1-p0.25"
    param_remode = "pixel"
    param_recount = 1
    
    timm.data.auto_augment._RAND_TRANSFORMS = _RAND_TRANSFORMS
    timm.data.auto_augment._RAND_INCREASING_TRANSFORMS = _RAND_INCREASING_TRANSFORMS

    transform = create_transform(
        input_size=param_input_size,
        is_training=True,
        color_jitter=None,
        auto_augment=param_aa,
        interpolation='bicubic',
        scale=param_crop_scale,
        ratio=param_crop_aspect_ratio,
        re_prob=0,
        re_mode=param_remode,
        re_count=param_recount,
        mean=mean,
        std=std,
        hflip=0,
    )
    # removing random erasing
    del transform.transforms[-1]
    # build ours random erasing
    re_transform = RandomErasing(
        param_reprob, max_area=param_rearea, mode=param_remode, max_count=param_recount, device='cpu'
    )

    transform.transforms.append(re_transform)

    if (overcrop_input_size_w is not None) or (overcrop_input_size_h is not None):
        assert (overcrop_input_size_w is not None) and (overcrop_input_size_h is not None)
        assert overcrop_input_size_w >= param_input_size
        assert overcrop_input_size_h >= param_input_size
        del transform.transforms[0]
        transform.transforms.insert(
            0,
            RandomResizedPaddingCrop(
                param_input_size,
                overcrop_input_h=overcrop_input_size_h,
                overcrop_input_w=overcrop_input_size_w,
                scale=param_crop_scale, ratio=param_crop_aspect_ratio,
                interpolation=Image.BICUBIC
            )
        )

    transform.transforms.insert(0, torchvision.transforms.RandomGrayscale(0.1))
    transform.transforms.insert(0, torchvision.transforms.Resize(param_aug_input_size))
    transform.transforms.insert(0, torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.02))

    if param_aa_geo is not None:
        geo_transform1 = build_geo_transform(param_input_size, param_aa_geo, ['Rotate', 'ShearX', 'ShearY'], mean)
        geo_transform2 = build_geo_transform(param_input_size, param_aa_geo_trans, ['TranslateXRel', 'TranslateYRel'], mean)
        transform.transforms.insert(-2, geo_transform1)
        transform.transforms.insert(-2, geo_transform2)
    print('building MAE style transform:{}'.format(transform))

    transform_te = torchvision.transforms.Compose([
        torchvision.transforms.Resize(param_aug_input_size),
        torchvision.transforms.CenterCrop(
            (overcrop_input_size_h, overcrop_input_size_w) if overcrop_input_size_w is not None else param_input_size
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    print(transform_te)

    return transform, transform_te

def auto_build_train_test_transform(
    transform_type,
    aug_input_size=242, input_size=224, scale=(0.78, 0.86), 
    gray_scale=False, random_skew=False
):
    
    if transform_type == "lux_transform":
        trans_tr, trans_te = build_train_test_lux_transform(
            aug_input_size=aug_input_size, input_size=input_size, scale=scale
        )
    elif transform_type == "lux_transform_with_overcrop":
        trans_tr, trans_te = build_train_test_lux_transform(
            aug_input_size=260, input_size=240, scale=scale, 
            overcrop_input_size_w=250, overcrop_input_size_h=250
        )
    elif transform_type == "classic_transform":
        trans_tr, trans_te = build_train_test_transform(
            aug_input_size=aug_input_size, input_size=input_size, scale=scale,
            gray_scale=gray_scale, random_skew=random_skew
        )
    elif transform_type == "classic_transform_with_overcrop":
        assert scale == (0.78, 0.86)
        trans_tr, trans_te = build_train_test_transform(
            aug_input_size=260, input_size=240, scale=scale,
            gray_scale=gray_scale, random_skew=random_skew,
            overcrop_input_size_w=250, overcrop_input_size_h=250
        )
    else:
        raise ValueError("undefined transform type {}".format(transform_type))
    
    return trans_tr, trans_te


class ImagelistDataset(object):
    def __init__(self, imglist, loader, transform, return_2samples=False):
        #self.imglist = imglist
        use_superclass = False
        if all([len(i) == 2 for i in imglist]):
            use_superclass = False
        elif all([len(i) == 3 for i in imglist]): 
            print("ImagelistDataset using superclass")
            use_superclass = True
        else:
            raise ValueError("imglist invalid")
        
        self.label_to_int = {
            label : idx
            for idx, label in enumerate(sorted(list(set([i[1] for i in imglist]))))
        }
        
        self.int_to_label = {
            idx : label
            for label, idx in self.label_to_int.items()
        }
        
        self.loader = loader
        self.transform = transform
        self.class_nums = len(self.label_to_int)

        self.class_to_superclass = None
        if use_superclass:
            self.superlabel_to_int = {
                label : idx
                for idx, label in enumerate(sorted(list(set([i[2] for i in imglist]))))
            }
            self.label_to_superlabel = {}
            for _, label, super_label in imglist:
                assert super_label in self.superlabel_to_int
                # a.jpg iot001-l iot001
                if label in self.label_to_superlabel:
                    assert self.label_to_superlabel[label] == super_label
                else:
                    self.label_to_superlabel[label] = super_label
            
            
            self.class_to_superclass = {
                int_label : self.superlabel_to_int[self.label_to_superlabel[label]]
                for int_label, label in self.int_to_label.items()
            }

        self.imglist = [i[:2] for i in imglist]
        self.label_to_indexes = {}
        for idx, (_, label) in enumerate(self.imglist):
            if label not in self.label_to_indexes:
                self.label_to_indexes[label] = []

            self.label_to_indexes[label].append(idx)

        self.return_2samples = return_2samples
        

    def __getitem__(self, index):
        impath, label = self.imglist[index]
        img = self.loader(impath)
        img = self.transform(img)
        target = self.label_to_int[label]

        if self.return_2samples:
            #indexes = self.label_to_indexes[label]
            assert index in self.label_to_indexes[label]
            #indexes = [i for i in indexes if i != index]
            if len(self.label_to_indexes[label]) == 1:
                img_another = img
            else:  
                index_another = index
                while index_another == index:
                    index_another = np.random.choice(self.label_to_indexes[label])
                
                impath_another, label_another = self.imglist[index_another]
                assert label_another == label

                img_another = self.loader(impath_another)
                img_another = self.transform(img_another)
            
            #superclass_target = self.class_to_superclass[target]
            return img, img_another, target #, superclass_target
        else:
            return img, target

    def __len__(self):
        return len(self.imglist)


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.list = [i for i in range(start, end)]

    def __iter__(self):
        return iter(self.list)

    def shuffle(self):
        np.random.shuffle(self.list)

class MyMapDataset(object):
    def __init__(self, start, end):
        super(MyMapDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.list = [i for i in range(start, end)]

    def __getitem__(self, index):
        return self.list[index]

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':

    ds = MyMapDataset(start=3, end=14)
    loader = torch.utils.data.DataLoader(ds, num_workers=8, batch_size=3, shuffle=True)
    
    for i in range(5):
        loader_it = iter(loader)
        cnt = 0
        while cnt < 2:
            try:
                print(next(loader_it))
            except Exception as e:
                print("shuffling")
                #loader.dataset.shuffle()
                loader_it = iter(loader)

            cnt += 1
            
    from IPython import embed
    embed()
