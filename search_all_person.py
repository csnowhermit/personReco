import argparse
import time
from sys import platform
from collections import Counter
import subprocess

from models import *
from utils.datasets import *
from utils.utils import *
from utils.channel import Channel
from utils.Logger import Logger
from utils.rangeJudgeUtil import isInsidePolygon

from reid.data import make_data_loader, splitDistmat, make_pidNames_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg

'''
    cv2的颜色为BGR，matplotlib为RGB
    识别所有人：包括乘客
'''

# # 将识别结果写入到ffmpeg流（推流）
# rtsp = "rtsp_address"
# rtmp = "rtmp://localhost:1935/live/home"
#
# size = (1280, 720)
# sizeStr = str(size[0]) + 'x' + str(size[1])
#
# command = ['ffmpeg',
#            '-y', '-an',
#            '-f', 'rawvideo',
#            '-vcodec', 'rawvideo',
#            '-pix_fmt', 'bgr24',
#            '-s', sizeStr,
#            '-r', '25',
#            '-i', '-',
#            '-c:v', 'libx264',
#            '-pix_fmt', 'yuv420p',
#            '-preset', 'ultrafast',
#            '-f', 'flv',
#            rtmp]
#
# pipe = subprocess.Popen(command
#                         , shell=False
#                         , stdin=subprocess.PIPE
#                         )

# # 生产者端，写gui要的数据到channel
# pub = Channel()

# 日志
# log = Logger('D:/logs/person-search/search_all_person.log', level='info')

fo = open("D:/logs/person-search/search_all_person.txt", 'w+', encoding="utf-8")

# 车厢地面坐标范围（该范围外的人为车厢外的人），摄像头在右侧（120_150_2.mp4文件用）
# polyList = [{'x': 620, 'y': 120}, {'x': 740, 'y': 120}, {'x': 60, 'y': 720}, {'x': 1040, 'y': 720}]    # 车厢两端摄像头
# polyList = [{'x': 400, 'y': 200}, {'x': 800, 'y': 200}, {'x': 60, 'y': 720}, {'x': 1040, 'y': 720}]    # 车厢中部摄像头
# # 摄像头在左侧
polyList = [{'x': 423, 'y': 0}, {'x': 606, 'y': 0}, {'x': 140, 'y': 720}, {'x': 1070, 'y': 720}]
# 全面积（只检测本节车厢的，摄像头在中间：220；摄像头在两边：109）
# polyList = [{'x': 200, 'y': 180}, {'x': 335, 'y': 125},
#             {'x': 480, 'y': 180}, {'x': 650, 'y': 180},
#             {'x': 750, 'y': 100}, {'x': 1000, 'y': 720},
#             {'x': 0, 'y': 720}, {'x': 0, 'y': 336}]

def detect(cfg,
           data,
           weights,
           images='data/samples',  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.5,
           dist_thres=1.0,
           save_txt=False,
           save_images=True):

    # Initialize
    device = torch_utils.select_device(force_cpu=False)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    ############# 加载<pid, 中文名, 颜色>列表 #############
    pidNameInfo = make_pidNames_loader(reidCfg)
    print("pidNameInfo:", pidNameInfo)

    ############# 行人重识别模型初始化 #############
    query_loader, num_query = make_data_loader(reidCfg)    # 迭代器，待查样本数
    print("query_loader:", type(query_loader), query_loader)

    reidModel = build_model(reidCfg, num_classes=10126)    # 行人重识别使用ResNet50网络
    reidModel.load_param(reidCfg.TEST.WEIGHT)
    reidModel.to(device).eval()

    query_feats = []
    query_pids  = []

    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch    # 图像，行人ID，相机ID

            img = img.to(device)
            feat = reidModel(img)         # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
            print("feat:", type(feat), feat.shape, feat)
            query_feats.append(feat)
            query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2) # 计算出查询图片的特征向量

    ############# 行人检测模型初始化 #############
    model = Darknet(cfg, img_size)    # yolov3使用Darknet53网络

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()
    # Half precision
    opt.half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
    if opt.half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if opt.webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size, half=opt.half)
    else:
        dataloader = LoadImages(images, img_size=img_size, half=opt.half)

    # Get classes and colors
    # parse_data_cfg(data)['names']:得到类别名称文件路径 names=data/coco.names
    classes = load_classes(parse_data_cfg(data)['names']) # 得到类别名列表: ['person', 'bicycle'...]
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))] # 对于每种类别随机使用一种颜色画框
    # colors = [[40, 92, 230] for _ in range(len(classes))]  # 只检测人，使用相同的颜色

    # Run inference
    t0 = time.time()

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        personNumsDict = {}  # <人员类型，人数>
        personNumsDict['总人数'] = 0    # 初始化persons为0
        for k in pidNameInfo.keys():    # 初始化各种人数为0
            personNumsDict[pidNameInfo[str(k)][1]] = 0

        t = time.time()
        # if i < 500 or i % 5 == 0:
        #     continue
        save_path = str(Path(output) / Path(path).name) # 保存的路径

        # Get detections shape: (3, 416, 320)
        img = torch.from_numpy(img).unsqueeze(0).to(device) # torch.Size([1, 3, 416, 320])
        pred, _ = model(img) # 经过处理的网络预测，和原始的
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0] # torch.Size([5, 7])。non_max_suppression，nms，非极大值抑制

        # det为目标检测的所有结果
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size 映射到原图
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen image 1/3 data\samples\000493.jpg: 288x416 5 persons, Done. (0.869s)
            print('%gx%g ' % img.shape[2:], end='')  # print image size '288x416'
            for c in det[:, -1].unique():   # 对图片的所有类进行遍历循环
                n = (det[:, -1] == c).sum() # 得到了当前类别的个数，也可以用来统计数目
                if classes[int(c)] == 'person':
                    print('%g %ss' % (n, classes[int(c)]), end=', ') # 打印个数和类别'5 persons'

            # Draw bounding boxes and labels of detections
            # (x1y1x2y2, obj_conf, class_conf, class_pred)
            count = 0
            gallery_img = []
            gallery_loc = []
            for *xyxy, conf, cls_conf, cls in det: # 对于最后的预测框进行遍历
                # *xyxy: 对于原图来说的左上角右下角坐标: [tensor(349.), tensor(26.), tensor(468.), tensor(341.)]
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf) # 'person 1.00'
                if classes[int(cls)] == 'person':
                    #plot_one_bo x(xyxy, im0, label=label, color=colors[int(cls)])
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    w = xmax - xmin # 233
                    h = ymax - ymin # 602

                    # 这里检测在不在划定范围内（将车门外、车窗外、其他车厢的乘客过滤出去）（向下摄像头不使用该区域限制）
                    ptList = []
                    ptList.append({'x': xmin, 'y': ymin})    # 左上
                    ptList.append({'x': xmax, 'y': ymin})    # 右上
                    ptList.append({'x': xmin, 'y': ymax})    # 左下
                    ptList.append({'x': xmax, 'y': ymax})    # 右下
                    if in_carriage(ptList, polyList) is False:    # 如果不在指定车厢范围，则剔除掉这个人，总人数-1
                        n = n - 1
                        continue

                    # 如果检测到的行人太小了，感觉意义也不大
                    # 这里需要根据实际情况稍微设置下
                    if w*h > 500:
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        crop_img = im0[ymin:ymax, xmin:xmax] # HWC (602, 233, 3)
                        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                        crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                        gallery_img.append(crop_img)

            if gallery_img:
                gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                gallery_img = gallery_img.to(device)
                gallery_feats = reidModel(gallery_img) # torch.Size([7, 2048])
                print("The gallery feature is normalized")
                gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                # m: 2，待查询人的个数
                # n: 7，yolo检测到人的个数
                m, n = query_feats.shape[0], gallery_feats.shape[0]

                ################### 开始统计总人数 ###################
                if '总人数' not in personNumsDict.keys():
                    personNumsDict['总人数'] = n    # 总人数n
                else:
                    personNumsDict['总人数'] = personNumsDict['总人数'] + n
                ################### 统计总人数结束 ###################

                # print("query_feats.shape:", query_feats.shape, "gallery_feats.shape:", gallery_feats.shape)
                distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()    # .t()，矩阵转置
                # out=(beta∗M)+(alpha∗mat1@mat2)
                # qf^2 + gf^2 - 2 * qf@gf.t()
                # distmat - 2 * qf@gf.t()
                # distmat: qf^2 + gf^2
                # qf: torch.Size([2, 2048])
                # gf: torch.Size([7, 2048])
                distmat.addmm_(1, -2, query_feats, gallery_feats.t())   # distmat行数：待检人数；列数：库图片中人数
                # distmat = (qf - gf)^2
                # distmat = np.array([[1.79536, 2.00926, 0.52790, 1.98851, 2.15138, 1.75929, 1.99410],
                #                     [1.78843, 1.96036, 0.53674, 1.98929, 1.99490, 1.84878, 1.98575]])
                distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)，cuda转到内存中
                print("distmat:", type(distmat), m, n, distmat.shape)

                pidMatDict = splitDistmat(query_pids, distmat)    # 按pid拆分，按行拆分，<pid, mat>
                pid_freq = Counter(query_pids)

                marked = []  # 已经标注过的（为防止再次标注：）
                for k in pidMatDict.keys():    # k表示pid，对于每个人进行识别
                    tmp_mat = pidMatDict[k]  # 某一个人的矩阵
                    tmp_mat = tmp_mat.sum(axis=0) / pid_freq[k]  # 平均一下query中同一行人的多个结果

                    # tmp_mat = tmp_mat.sum(axis=0)
                    index = tmp_mat.argmin()  # 返回距离最小值的下标（图中哪个人最像待检测的人）
                    # print("tmp_mat:", type(tmp_mat), m, n, tmp_mat.shape, tmp_mat)    # <class 'numpy.ndarray'>，待检图片数m，涂上识别到人数n
                    print('距 离：%s' % tmp_mat[index], index, dist_thres, n)

                    # print("gallery_loc:", gallery_loc)
                    for index in range(n):
                        if tmp_mat[index] < dist_thres:
                            ################### 先统计该类型人员的人数 ###################
                            if str(k) not in personNumsDict.keys():
                                personNumsDict[pidNameInfo[str(k)][1]] = 1    # <站务（人员类别）, 人数>
                            else:
                                personNumsDict[pidNameInfo[str(k)][1]] = personNumsDict[pidNameInfo[str(k)][1]] + 1
                            ################### 统计该类型人员的人数 结束 ###################

                            marked.append(index)    # 标注：某个人是特定的人，而非普通乘客
                            print('距离：%s' % tmp_mat[index])
                            # # cv2显示中文出现乱码，需转为PIL添加中文后再转回来。这样在方法中调用的话最后显示为颜色涂层filled，没有内容。
                            # # 解决办法：把画框和添加中文label过程直接放到这里来，避免函数调用式转写。（英文可以直接调用该方法，不会出现乱码）
                            # plot_one_box(gallery_loc[index], im0, label='%s:%s' % (pidNameInfo[str(k)][1], tmp_mat[index]), color=getColorArr(pidNameInfo[str(k)][2]))

                            # 准备plot_one_box()方法的各项参数
                            x = gallery_loc[index]  # 画框范围（左上、右下坐标）
                            # label = '%s:%s' % (pidNameInfo[str(k)][1], tmp_mat[index])    # 要标记的内容（中文）
                            label = '%s' % pidNameInfo[str(k)][1]
                            # label = 'person:%s' % (tmp_mat[index])
                            color = getColorArr(pidNameInfo[str(k)][2])  # 画框颜色

                            tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line thickness，线条厚度
                            color = color or [random.randint(0, 255) for _ in range(3)]  # 如果没指定颜色，则随机
                            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上xy，右下xy
                            cv2.rectangle(im0, c1, c2, color, thickness=tl)

                            if label:
                                # 对于内容框，c1为左下，c2为右上
                                tf = max(tl - 1, 1)  # font thickness，字体粗细
                                # t_size = cv2.getTextSize(label, fontFace=0, fontScale=tl / 3, thickness=tf)[0]    # fontFace=1时，((310, 22), 10)
                                # t_size = cv2.getTextSize(label, fontFace=1, fontScale=tl / 3, thickness=tf)[0]   # fontFace=1时，((156, 10), 6)

                                zh_cn_nums = get_zhcn_number(label)  # 中文的字数（一个中文字20个像素宽，一个英文字10个像素宽）
                                t_size = (20 * zh_cn_nums + 10 * (len(label) - zh_cn_nums), 22)
                                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 纵坐标，多减3目的是字上方稍留空
                                cv2.rectangle(im0, c1, c2, color, -1)  # filled
                                # print("t_size:", t_size, " c1:", c1, " c2:", c2)

                                # Draw a label with a name below the face
                                # cv2.rectangle(im0, c1, c2, (0, 0, 255), cv2.FILLED)
                                font = cv2.FONT_HERSHEY_DUPLEX

                                # 将CV2转为PIL，添加中文label后再转回来
                                pil_img = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(pil_img)
                                font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')
                                draw.text((c1[0], c1[1] - 20), label, (255, 255, 255), font=font)

                                im0 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # PIL转CV2
                            # cv2.imshow('person search', im0)
                            # cv2.waitKey()
                        else:  # 普通乘客
                            if index not in marked:
                                x = gallery_loc[index]  # 画框范围（左上、右下坐标）
                                # label = '%s:%s' % (pidNameInfo[str(k)][0], tmp_mat[index])    # 要标记的内容（中文）
                                label = '乘客'  # 乘客
                                color = [151, 73, 228]  # 画框颜色，用青色标识

                                tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line thickness，线条厚度
                                color = color or [random.randint(0, 255) for _ in range(3)]  # 如果没指定颜色，则随机
                                c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上xy，右下xy
                                cv2.rectangle(im0, c1, c2, color, thickness=tl)

                                if label:
                                    # 对于内容框，c1为左下，c2为右上
                                    tf = max(tl - 1, 1)  # font thickness，字体粗细
                                    # t_size = cv2.getTextSize(label, fontFace=0, fontScale=tl / 3, thickness=tf)[0]    # fontFace=1时，((310, 22), 10)
                                    # t_size = cv2.getTextSize(label, fontFace=1, fontScale=tl / 3, thickness=tf)[0]   # fontFace=1时，((156, 10), 6)

                                    zh_cn_nums = get_zhcn_number(label)  # 中文的字数（一个中文字20个像素宽，一个英文字10个像素宽）
                                    t_size = (20 * zh_cn_nums + 10 * (len(label) - zh_cn_nums), 22)
                                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 纵坐标，多减3目的是字上方稍留空
                                    cv2.rectangle(im0, c1, c2, color, -1)  # filled
                                    # print("t_size:", t_size, " c1:", c1, " c2:", c2)

                                    # Draw a label with a name below the face
                                    # cv2.rectangle(im0, c1, c2, (0, 0, 255), cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX

                                    # 将CV2转为PIL，添加中文label后再转回来
                                    pil_img = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                                    draw = ImageDraw.Draw(pil_img)
                                    font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')
                                    draw.text((c1[0], c1[1] - 20), label, (255, 255, 255), font=font)

                                    im0 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # PIL转CV2
                                # cv2.imshow('person search', im0)
                                # cv2.waitKey()
        ##############################
        # pipe.stdin.write(im0.tostring())    # 写入到rtmp（识别过程太慢，导致ffplay拉取不到）
        # 替代做法：使用本地gui，将(personNumsDict, im0)写入到队列，gui端直接拉取
        # lock.acquire()
        # frame_buffer.push(im0)
        # lock.release()
        # yield personNumsDict, im0
        # pub.publish(pub.channel_name, personNumsDict)
        # pub.publish(pub.channel_name, im0)

        # print("personNumsDict:", personNumsDict)
        # log.logger.info(personNumsDict)
        fo.write(str(personNumsDict) + "\n")
        print('Done. (%.3fs)' % (time.time() - t))

        if opt.webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)
    print('Done. (%.3fs)' % (time.time() - t0))


'''
     '[40, 92, 230]'转 [40, 92, 230]
'''
def getColorArr(colorStr):
    rgbList = []
    if colorStr is not None or len(colorStr) > 0:
        colorStr = colorStr.replace("[", "").replace("]", "")
        arr = colorStr.split(",")
        for a in arr:
            rgbList.append(int(a))
    return rgbList

'''
    判断字符串中中文的个数
'''
def get_zhcn_number(s):
    count = 0
    for item in s:
        if 0x4E00 <= ord(item) <= 0x9FA5:
            count += 1
    return count

'''
    判断在不在车厢里（避免把门外和窗外的人识别到）
    判断标准：左下(xmin, ymax)和右下(xmax, ymax)角在车厢里，就认为在车厢里
    :param pointList 框人的四点坐标
    :param polyList 车厢各点坐标
    :return True，在车厢里；False，不在车厢里
'''
def in_carriage(pointList, polyList):
    for pt in pointList:
        flag = isInsidePolygon(pt, polyList)    # 在范围内，返回True
        if flag is True:    # 只要有一个点在车厢里，就认为在车厢里
            return True
    print("不在范围内的原因：", pointList)
    return False

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

    with torch.no_grad():
        detect(opt.cfg,
               opt.data,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               dist_thres=opt.dist_thres,
               fourcc=opt.fourcc,
               output=opt.output)