# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
from torch.utils.data import DataLoader

from .collate_batch import  val_collate_fn
from .datasets import ImageDataset
from .transforms import build_transforms
from .datasets import Market1501

def make_data_loader(cfg):
    # 验证集的预处理
    val_transforms = build_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS # 加载图像进程数 8
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)

    val_set = ImageDataset(dataset.query, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return val_loader, len(dataset.query)

'''
    按行拆分相似度矩阵，每个人单独识别
'''
def splitDistmat(query_pids, distmat):
    dict = {}
    for i in range(len(query_pids)):
        if query_pids[i] in dict.keys():
            dict[query_pids[i]] = np.append(dict[query_pids[i]], distmat[i, ])
        else:
            dict[query_pids[i]] = distmat[i, ]
    for k in dict.keys():
        dict[k] = dict[k].reshape(-1, distmat.shape[1])    # reshape为x行n列，x为同一个人的照片个数
    return dict

'''
    加载“personID, 姓名, 人员类型, 颜色”列表
    :return Dict <personID, (姓名, 人员类型，颜色)>
'''
def make_pidNames_loader(cfg):
    pidNameInfo = {}
    with open(cfg.PID_NAMES_FILE, 'r', encoding="utf-8") as fo:
        for line in fo.readlines():
            arr = line.strip("\n").split("@")
            if len(arr) == 3:
                pidNameInfo[str(arr[0])] = (arr[1], arr[2], [100, 100, 100])    # 默认标记颜色为灰色
            else:
                pidNameInfo[str(arr[0])] = (arr[1], arr[2], arr[3])
    return pidNameInfo
