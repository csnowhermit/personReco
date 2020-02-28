import argparse
import time
from sys import platform
from collections import Counter
import subprocess
import threading

from models import *
from utils.datasets import *
from utils.utils import *
from utils.capUtil import Stack

from reid.data import make_data_loader, splitDistmat, make_pidNames_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg

def read_video(opt,
               frame_buffer,
               images='data/samples',
               img_size=416):

    # Set Dataloader
    if opt.webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size, half=opt.half)
    else:
        dataloader = LoadImages(images, img_size=img_size, half=opt.half)

    num = 0
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        # frame_buffer.push((path, img, im0, vid_cap))
        if str(path).endswith(".mp4"):
            frame_buffer.append(("%s_%04d.jpeg" % (str(path), num), str(img), str(im0), str(vid_cap)))
            num = num + 1
        else:
            frame_buffer.append((str(path), str(img), str(im0), str(vid_cap)))

    with open("./testData/frame_buffer.txt", 'a+', encoding="utf-8") as fo:
        for path, img, im0, vid_cap in frame_buffer:
            fo.writelines((str(path), " | ", str(img), " | ", str(im0), " | ", str(vid_cap), " \n"))
            fo.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help="模型配置文件路径")
    parser.add_argument('--data', type=str, default='data/coco.data', help="数据集配置文件所在路径")
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='模型权重文件路径')
    parser.add_argument('--images', type=str, default='data/samples', help='需要进行检测的图片文件夹')
    parser.add_argument('-q', '--query', default=r'query', help='查询图片的读取路径.')
    parser.add_argument('--img-size', type=int, default=416, help='输入分辨率大小')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='物体置信度阈值')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS阈值')
    parser.add_argument('--dist_thres', type=float, default=1.0, help='行人图片距离阈值，小于这个距离，就认为是该行人')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='检测后的图片或视频保存的路径')
    parser.add_argument('--half', default=False, help='是否采用半精度FP16进行推理')
    parser.add_argument('--webcam', default=False, help='是否使用摄像头进行检测')
    opt = parser.parse_args()
    print(opt)

    # 准备队列
    # frame_buffer = Stack(3)  # 3帧3帧地过
    frame_buffer = []
    read_video(opt,
               frame_buffer,
               opt.images,
               416)