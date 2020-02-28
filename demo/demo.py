
import numpy as np
import torch
from torch.utils.data import DataLoader

from reid.data.collate_batch import  val_collate_fn
from reid.data.datasets import ImageDataset
from reid.data.transforms import build_transforms
from reid.data.datasets import Market1501

from reid.config import cfg as reidCfg


def load_img(cfg):
    # 验证集的预处理
    val_transforms = build_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS  # 加载图像进程数 8
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)
    print("dataset:", type(dataset), dataset)
    print("dataset.query:", type(dataset.query), dataset.query)

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


if __name__ == '__main__':
    # val_loader, query_num = load_img(reidCfg)
    # print("val_loader:", type(val_loader), val_loader)
    # print("query_num：", type(query_num), query_num)
    # a = torch.randn(2, 3)  # 标准正态分布生成随机数，2行3列
    # print("a:", type(a), a.shape, a)
    # b = a.numpy()
    # print("b:", type(b), b.shape, b)
    # c = b.sum(axis=0) / b.shape[0]
    # print("c:", c)
 #    distmat = [[2.0106, 1.9585,  1.992,  1.992, 1.9951, 1.9743, 2.1014, 2.0987, 1.9439, 1.9589, 2.0725, 2.0935, 1.9696, 1.7445,  1.881, 1.7478, 2.0981, 1.9938, 1.7843, 1.9985,  1.789],
 # [2.0304, 1.9236,  1.943, 2.0222, 1.9519, 1.8986, 2.0825, 2.0381, 1.9963, 1.9021, 2.0337, 2.0731, 1.9016, 1.7721, 1.8662, 1.6568, 2.0704, 1.9824, 1.7921,   1.95,  1.734],
 # [2.0495, 1.9315, 1.9673, 2.0502, 1.9731, 1.9221, 2.1024, 2.0116, 2.0153, 1.9368,  2.021, 2.0766, 1.9627, 1.7521, 1.8613,  1.717,  2.071,  2.017, 1.7856, 1.9896, 1.7376],
 # [1.9501, 2.0109, 2.0829, 2.0024, 1.8812,  2.099, 1.7313, 1.8056, 2.0123, 2.0981, 1.9454, 2.0865, 1.8207, 2.0998, 1.8734, 1.8324,  1.956, 1.9304,  2.017, 1.8827, 2.0244],
 # [2.0402, 1.9769, 2.0958, 2.0089, 1.9467, 2.0908, 1.7761, 1.9389, 2.0196, 2.0584, 1.9742, 2.1092, 1.8695, 2.0605, 1.9623, 1.8432, 1.9605, 1.9812, 2.0181, 1.8957,   2.02],
 # [1.9837, 1.9855, 2.0506, 2.0072, 1.8787, 2.0878, 1.7274, 1.9527, 1.9985,  2.069, 1.9334, 2.1421, 1.9342, 2.0683, 2.0037, 1.8986, 1.9888, 2.0065, 2.0416, 1.8576, 2.0303]]
    distmat = [[1.7796, 0.93217, 2.0237],
               [1.771, 0.8506, 1.9666],
               [1.7793, 0.89047, 2.0116],
               [0.23064, 1.7526, 1.811],
               [0.29268, 1.7386, 1.8748],
               [0.39406, 1.823, 1.8822]]
    distmat = np.asarray(distmat)
    query_pids = [1, 1, 1, 2, 2, 2]

    # 对query_pid统计每个元素出现的个数（每个人的照片数）
    from collections import Counter
    pid_freq = Counter(query_pids)
    print("pid_freq:", type(pid_freq), pid_freq)

    dict = splitDistmat(query_pids, distmat)
    for k in dict.keys():    # 对于每一个人分别计算平均相似度
        tmp_mat = dict[k]    # 某一个人的矩阵
        print("person", k, "的相似度矩阵：", type(tmp_mat), tmp_mat.shape)
        print(tmp_mat)

        tmp_mat = tmp_mat.sum(axis=0) / pid_freq[k]  # 平均一下query中同一行人的多个结果
        print("平均完之后：")
        print(type(tmp_mat), tmp_mat)

        # distmat = distmat.sum(axis=0)
        index = tmp_mat.argmin()  # 返回距离最小值的下标（图中哪个人最像待检测的人）
        # print("distmat:", type(distmat), m, n, distmat.shape, distmat)
        print('距 离：%s' % tmp_mat[index], index)

        # plot_one_box(gallery_loc[index], im0, label='find! %s' % distmat[index], color=colors[int(cls)])

