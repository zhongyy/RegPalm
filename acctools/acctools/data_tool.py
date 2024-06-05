import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from . import util
import threading
import shutil

THREAD_LOCK = threading.Lock()

def read_csv(csv_file, split_char=' ', drop_head=False):
    with open(csv_file, "r") as f:
        lines = f.readlines()

    if drop_head:
        lines = lines[1:]

    return [i.strip()[1:-1].split('"{}"'.format(split_char)) for i in lines]

def readlines(filename):
    with open(filename, "r") as f:
        return f.readlines()

def writelines(filename, lines):
    with open(filename, "w") as f:
        util.TYPE_CHECK(lines, list)
        print("Write to file using happytools ...")
        for line in tqdm(lines):
            f.write("{}\n".format(line))

def generate_file_list(data_root, pattern=["*"], save_path=None, label_fn=None, abs_path=False, sort=False):
    r""" This function generates file list.

    Arguments:
        data_root (string, required): path to data root
        pattern (list of strings, optional): file pattern, such as ["*.jpg", "*.png"]
        save_path (string, optional): path to save the list
        label_fn (function, optional): a function extract label string from the (relative) image path.
            when images are saved in $root/xxx/n.jpg (where xxx is label string),
            label_fn("xxx/1.jpg") -> "xxx"
    """
    if not os.path.exists(data_root):
        raise ValueError("{} not exists".format(data_root))

    path = Path(data_root)
    result = []
    if not type(pattern) is list:
        pattern = [pattern]

    for pat in pattern:
        if abs_path is False: 
            result.extend([str(i) for i in path.rglob(pat) if os.path.isfile(str(i))])
        else:
            result.extend([os.path.abspath(str(i)) for i in path.rglob(pat) if os.path.isfile(str(i))])


    if sort:
        result = sorted(result)

    label_dict = {}
    
    # generate label aside with img path
    for i in range(len(result)):
        if label_fn:
            label = label_fn(result[i])
            label_dict[label] = 0
            result[i] = "{} {}\n".format(result[i], label)
        else:
            result[i] = "{}\n".format(result[i])
    
    if len(result) < 1:
        raise ValueError("no imgs at all at {}".format(data_root))


    print("In root {}, {} files are found, {} classes in total".format(
        os.path.abspath(data_root), len(result), len(label_dict.keys()))) 

    if save_path:
        with open(save_path, 'w') as f:
            for i in result:
                f.write(i)
        
        print("writen in {}".format(os.path.abspath(save_path)))

    return result

def rm_r(path):
    assert os.path.exists(path)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)



            
