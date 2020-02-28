
import os
import cv2
from utils.utils import *


'''
    手动标注矩形框
'''

pointList = [[571, 418, 677, 516],
             [408, 190, 480, 242]]

im0 = cv2.imread("D:/logs/loss.jpg")
color = [random.randint(0, 255) for _ in range(3)]  # 如果没指定颜色，则随机

for point in pointList:
    x = point

    label = "handbag 0.98"
    # plot_one_box(x, im0, label=label)
    tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line thickness，线条厚度

    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上xy，右下xy
    cv2.rectangle(im0, c1, c2, color, thickness=tl)

    if label:
        # 对于内容框，c1为左下，c2为右上
        tf = max(tl - 1, 1)  # font thickness，字体粗细
        # t_size = cv2.getTextSize(label, fontFace=0, fontScale=tl / 3, thickness=tf)[0]    # fontFace=1时，((310, 22), 10)
        # t_size = cv2.getTextSize(label, fontFace=1, fontScale=tl / 3, thickness=tf)[0]   # fontFace=1时，((156, 10), 6)

        # zh_cn_nums = get_zhcn_number(label)  # 中文的字数（一个中文字20个像素宽，一个英文字10个像素宽）
        # t_size = (20 * zh_cn_nums + 10 * (len(label) - zh_cn_nums), 22)
        t_size = (10 * len(label), 22)
        # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
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


cv2.imwrite("D:/logs/loss_rect.jpg", im0)


# tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line thickness，线条厚度
# color = color or [random.randint(0, 255) for _ in range(3)]  # 如果没指定颜色，则随机
# c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上xy，右下xy
# cv2.rectangle(im0, c1, c2, color, thickness=tl)


# 对于内容框，c1为左下，c2为右上
# tf = max(tl - 1, 1)  # font thickness，字体粗细
# # t_size = cv2.getTextSize(label, fontFace=0, fontScale=tl / 3, thickness=tf)[0]    # fontFace=1时，((310, 22), 10)
# # t_size = cv2.getTextSize(label, fontFace=1, fontScale=tl / 3, thickness=tf)[0]   # fontFace=1时，((156, 10), 6)

# zh_cn_nums = get_zhcn_number(label)  # 中文的字数（一个中文字20个像素宽，一个英文字10个像素宽）
# t_size = (20 * zh_cn_nums + 10 * (len(label) - zh_cn_nums), 22)
# c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 纵坐标，多减3目的是字上方稍留空
# cv2.rectangle(im0, c1, c2, color, -1)  # filled
# # print("t_size:", t_size, " c1:", c1, " c2:", c2)

# # Draw a label with a name below the face
# # cv2.rectangle(im0, c1, c2, (0, 0, 255), cv2.FILLED)
# font = cv2.FONT_HERSHEY_DUPLEX

# # 将CV2转为PIL，添加中文label后再转回来
# pil_img = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
# draw = ImageDraw.Draw(pil_img)
# font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')
# draw.text((c1[0], c1[1] - 20), label, (255, 255, 255), font=font)

# im0 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # PIL转CV2
