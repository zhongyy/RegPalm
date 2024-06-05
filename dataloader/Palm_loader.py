import numpy as np
import scipy.misc
import os
import torch
from PIL import Image
from io import BytesIO
import base64
import torchvision

from .common_data import ImagelistDataset, auto_build_train_test_transform

def _readlines(filename):
    with open(filename, "r") as f:
        return f.readlines()

_IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def _is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in _IMG_EXTENSIONS)

def _pil_loader(path):
    if _is_image_file(path):
        #if read_image_print:
        #    print("read image path")
        #    read_image_print = False
        if not os.path.exists(path):
            raise IOError('"{}" does not exist'.format(path))
        
        got_img = False
        while not got_img:
            try:
                img = Image.open(path).convert('RGB')
                got_img = True
            except IOError:
                print(
                    'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                    .format(path)
                )
    else:
        img = Image.open(BytesIO(base64.b64decode(path))).convert('RGB')

    return img

def get_MPD_train_and_test_dataset(mpd_imglist, transform_type, fold_idx, num_folds=5, 
    gray_scale=False, random_skew=False, use_superclass=False, return_2samples=False
):
    assert fold_idx < num_folds
    assert os.path.exists(mpd_imglist)
    assert type(use_superclass) is bool
    assert not use_superclass

    lines = [i.strip().split(" ") for i in _readlines(mpd_imglist)]
    assert all([len(i) == 2 for i in lines])

    label_to_idxes = {}
    for idx, (_, label) in enumerate(lines):
        if label not in label_to_idxes:
            label_to_idxes[label] = []

        label_to_idxes[label].append(idx)

    sorted_labels = sorted([i for i in label_to_idxes])
    num_slice = len(sorted_labels) // num_folds
    test_labels = {i : None for i in sorted_labels[num_slice * fold_idx : num_slice * (fold_idx + 1)]}
    train_labels = {i : None for i in sorted_labels if i not in test_labels}
    test_labels = sorted([i for i in test_labels])
    train_labels = sorted([i for i in train_labels])
    print("{} ({} ... {}) {} ({} ... {})".format(
        len(train_labels), train_labels[0], train_labels[-1],
        len(test_labels), test_labels[0], test_labels[-1]
    ))
    
    train_lines = []
    for label in train_labels:
        idxes = label_to_idxes[label]
        train_lines.extend([lines[i] for i in idxes])

    query_lines = []
    gallery_lines = []
    np.random.seed(1024)
    for label in test_labels:
        idxes = label_to_idxes[label]
        gallery_idx = np.random.choice(idxes)
        gallery_lines.append(lines[gallery_idx])
        query_lines.extend([lines[i] for i in idxes if i != gallery_idx])
    
    trans_tr, trans_te = auto_build_train_test_transform(
        transform_type=transform_type,
        aug_input_size=242, input_size=224, scale=(0.78, 0.86), 
        gray_scale=gray_scale, random_skew=random_skew
    )
    
    train_set = ImagelistDataset(train_lines, _pil_loader, transform=trans_tr, return_2samples=return_2samples)
    query_set = ImagelistDataset(query_lines, _pil_loader, transform=trans_te)
    gallery_set = ImagelistDataset(gallery_lines, _pil_loader, transform=trans_te)

    return train_set, query_set, gallery_set

def get_SelfDefine_train_and_test_dataset(train_imglist, test_imglist, transform_type, 
    gray_scale=False, random_skew=False, use_superclass=False, return_2samples=False
):
    assert os.path.exists(train_imglist)
    lines = [i.strip().split(" ") for i in _readlines(train_imglist)]
    assert type(use_superclass) is bool
    if use_superclass:
        assert all([len(i) == 3 for i in lines])
        for _, label, super_label in lines:
            assert label.endswith("-l") or label.endswith("-r"), label
            assert label[:-2] == super_label

        assert all([i[:] for i in lines])
    else:
        assert all([len(i) == 2 for i in lines])

    label_to_idxes = {}
    for idx, line in enumerate(lines):
        label = line[1]
        if label not in label_to_idxes:
            label_to_idxes[label] = []

        label_to_idxes[label].append(idx)

    labels = sorted([i for i in label_to_idxes])
    print("training: {} ids".format(len(labels)))
    train_lines = []
    for label in labels:
        idxes = label_to_idxes[label]
        train_lines.extend([lines[i] for i in idxes])


    assert os.path.exists(test_imglist)
    lines = [i.strip().split(" ") for i in _readlines(test_imglist)]
    if use_superclass:
        assert all([len(i) == 3 for i in lines])
        for _, label, super_label in lines:
            assert label.endswith("-l") or label.endswith("-r"), label
            assert label[:-2] == super_label

        assert all([i[:] for i in lines])
    else:
        assert all([len(i) == 2 for i in lines])

    label_to_idxes = {}
    for idx, line in enumerate(lines):
        label = line[1]
        if label not in label_to_idxes:
            label_to_idxes[label] = []

        label_to_idxes[label].append(idx)

    labels = sorted([i for i in label_to_idxes])
    print("test: {} ids".format(len(labels)))

    query_lines = []
    gallery_lines = []
    np.random.seed(1024)
    for label in labels:
        idxes = label_to_idxes[label]
        gallery_idx = np.random.choice(idxes)
        gallery_lines.append(lines[gallery_idx])
        query_lines.extend([lines[i] for i in idxes if i != gallery_idx])
    
    trans_tr, trans_te = auto_build_train_test_transform(
        transform_type=transform_type,
        aug_input_size=242, input_size=224, scale=(0.78, 0.86), 
        gray_scale=gray_scale, random_skew=random_skew
    )

    train_set = ImagelistDataset(train_lines, _pil_loader, transform=trans_tr, return_2samples=return_2samples)
    query_set = ImagelistDataset(query_lines, _pil_loader, transform=trans_te)
    gallery_set = ImagelistDataset(gallery_lines, _pil_loader, transform=trans_te)

    return train_set, query_set, gallery_set

def get_simple_test_dataset(imglist, gray_scale=False, transform_type="classic_transform"):
    assert os.path.exists(imglist)
    lines = [i.strip().split(" ")[:2] for i in _readlines(imglist)]
    assert all([len(i) == 2 for i in lines])
    print("test: {} ids".format(len(set([i[1] for i in lines]))))
    
    _, trans_te = auto_build_train_test_transform(
        transform_type=transform_type,
        aug_input_size=242, input_size=224, scale=(0.78, 0.86), 
        gray_scale=gray_scale
    )
    data_set = ImagelistDataset(lines, _pil_loader, transform=trans_te)
   
    return data_set

def get_closeset_train_and_test_dataset(imglist, transform_type, fold_idx, num_folds=5, 
    gray_scale=False, random_skew=False, use_superclass=False, return_2samples=False
):
    assert fold_idx < num_folds
    assert os.path.exists(imglist)
    assert type(use_superclass) is bool
    assert not use_superclass

    lines = [i.strip().split(" ") for i in _readlines(imglist)]
    assert all([len(i) == 2 for i in lines])

    label_to_idxes = {}
    for idx, (_, label) in enumerate(lines):
        if label not in label_to_idxes:
            label_to_idxes[label] = []

        label_to_idxes[label].append(idx)

    sorted_labels = sorted([i for i in label_to_idxes])
    train_lines = []
    query_lines = []
    gallery_lines = []
    np.random.seed(1024)
    for label in sorted_labels:
        idxes = label_to_idxes[label]
        np.random.shuffle(idxes)
        if len(idxes) < num_folds * 2:
            train_lines.extend([lines[i] for i in idxes])
        else:
            num_slice = len(idxes) // num_folds
            assert num_slice >= 2
            test_idxes = {i : None for i in idxes[num_slice * fold_idx : num_slice * (fold_idx + 1)]}
            train_idxes = {i : None for i in idxes if i not in test_idxes}
            test_idxes = sorted([i for i in test_idxes])
            train_idxes = sorted([i for i in train_idxes])
            """
            print("{} ({} ... {}) {} ({} ... {})".format(
                len(train_idxes), train_idxes[0], train_idxes[-1],
                len(test_idxes), test_idxes[0], test_idxes[-1]
            ))
            """
            train_lines.extend([lines[i] for i in train_idxes])
            gallery_idx = np.random.choice(test_idxes)
            gallery_lines.append(lines[gallery_idx])
            query_lines.extend([lines[i] for i in test_idxes if i != gallery_idx])
    
    trans_tr, trans_te = auto_build_train_test_transform(
        transform_type=transform_type,
        aug_input_size=242, input_size=224, scale=(0.78, 0.86), 
        gray_scale=gray_scale, random_skew=random_skew
    )

    train_set = ImagelistDataset(train_lines, _pil_loader, transform=trans_tr, return_2samples=return_2samples)
    query_set = ImagelistDataset(query_lines, _pil_loader, transform=trans_te)
    gallery_set = ImagelistDataset(gallery_lines, _pil_loader, transform=trans_te)

    return train_set, query_set, gallery_set

def _get_auxiliary_train_dataset(imglist, aug_input_size, input_size, scale, transform_type, gray_scale, random_skew, use_superclass, return_2samples):
    assert os.path.exists(imglist)
    lines = [i.strip().split(" ") for i in _readlines(imglist)]
    assert type(use_superclass) is bool
    if use_superclass:
        assert all([len(i) == 3 for i in lines])
        for _, label, super_label in lines:
            assert label.endswith("-l") or label.endswith("-r"), label
            assert label[:-2] == super_label

        assert all([i[:] for i in lines])
    else:
        assert all([len(i) == 2 for i in lines])

    print("auxiliary: {} ids".format(len(set([i[1] for i in lines]))))
    if use_superclass:
        print("auxiliary: {} super ids".format(len(set([i[2] for i in lines]))))

    trans_tr, _ = auto_build_train_test_transform(
        transform_type=transform_type,
        aug_input_size=aug_input_size, input_size=input_size, scale=scale, 
        gray_scale=gray_scale, random_skew=random_skew
    )
    
    train_set = ImagelistDataset(lines, _pil_loader, transform=trans_tr, return_2samples=return_2samples)
    return train_set
 
def get_auxiliary_icdm_dataset(imglist, transform_type, gray_scale=False, random_skew=False, use_superclass=False, return_2samples=False):
    return _get_auxiliary_train_dataset(
        imglist, aug_input_size=224, input_size=224, scale=(0.92, 1.0), 
        transform_type=transform_type,
        gray_scale=gray_scale, random_skew=random_skew, 
        use_superclass=use_superclass,
        return_2samples=return_2samples
    )

def get_auxiliary_w270h260_dataset(imglist, transform_type, gray_scale=False, random_skew=False, use_superclass=False, return_2samples=False):
    return _get_auxiliary_train_dataset(
        imglist, aug_input_size=242, input_size=224, scale=(0.78, 0.86),
        transform_type=transform_type,
        gray_scale=gray_scale, random_skew=random_skew, 
        use_superclass=use_superclass,
        return_2samples=return_2samples
    )


if __name__ == '__main__':
    from IPython import embed
    mpd_list = "/mnt/datasets/MPD/mpd_all.txt"
    tr_set, q_set, g_set = get_closeset_train_and_test_dataset(mpd_list, 4)
    
    trainloader = torch.utils.data.DataLoader(tr_set, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    print(len(tr_set))
    
    for data in trainloader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[0])
        print(data[1])
        embed()
        exit()

