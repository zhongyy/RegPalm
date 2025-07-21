# RegPalm
Code for "RegPalm: Towards Large-Scale Palmprint Recognition by Reducing Pattern Variance"


## Prepare 

```
cd acctools
pip install . -U
```

## Train

```
python train.py --config $CONFIG
```

## Test

```
python test.py \
    --query ${QUERY_LIST} \
    --gallery ${GALLERY_LIST} \
    --fp ${FP_LIST} \
    --model ${MODEL_NAME} \
    --pth ${PTH} \
    --device ${DEVICE} \
    --save ${SAVE_DIR} \
    --group
```

## Dataset 
Please visit https://zhongyy.github.io/WebPalm for details.

## Citation

If you find **RegPalm** useful in your research, please consider citing:

@article{zhong2025towards,
author    = {Zhong, Yaoyao and Chai, Weilong and Wang, Libin and Zheng, Dandan and Fu, Huiyuan and Ma, Huadong},
title     = {RegPalm: Towards Large-Scale Open-Set Palmprint Recognition by Reducing Pattern Variance},
journal   = {IEEE Transactions on Information Forensics and Security}, 
year      = {2025},
}
