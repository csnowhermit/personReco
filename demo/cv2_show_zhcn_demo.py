import cv2
import random
from PIL import Image, ImageFont, ImageDraw
import numpy as np

'''
    cv2显示中文Demo
    cv2.putText()，显示中文会出现乱码，解决方案：将img转为PIL，标注中文后在转回来
    注意：这种转换必须在调用方直接写才有效，封装个function进行调用的话会出现 标注区域会被涂上字体颜色，没文字内容
'''

# src=cv.imread('E:\imageload\example.png')
# cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
# cv.imshow('input_image', src)
# cv.waitKey(0)
# cv.destroyAllWindows()

'''
    画框，标注label
'''
def plot_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness，线条厚度
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))    # 左上xy，右下xy
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled

        # Draw a label with a name below the face
        # cv2.rectangle(img, c1, c2, (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        # 将CV2转为PIL，添加中文label后再转回来
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')
        draw.text((c1[0], c1[1]-20), label, (255, 255, 255), font=font)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # PIL转CV2
        # cv2.imshow("demo02", img)
        # cv2.waitKey()
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


im0 = cv2.imread("../testData/input/c1s1_123456.jpg")
print(type(im0), im0.shape)

gallery_loc = [(580, 28, 678, 280), (585, 6, 620, 99), (74, 47, 132, 279)]    # 图中人的位置
markList = [1, 0]    # 要标出哪些人

for m in markList:
    # plot_box(gallery_loc[m], im0, label="要找的人", color=None)
    label = "要找的人"    # 提前定义label
    x = gallery_loc[m]

    tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line thickness，线条厚度
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # 左上xy，右下xy
    cv2.rectangle(im0, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness，字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im0, c1, c2, color, -1)  # filled

        # Draw a label with a name below the face
        # cv2.rectangle(im0, c1, c2, (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        # 将CV2转为PIL，添加中文label后再转回来
        pil_img = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')
        draw.text((c1[0], c1[1] - 20), label, (255, 255, 255), font=font)

        im0 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # PIL转CV2

cv2.imshow("demo02", im0)
cv2.waitKey()

save_path = "D:/logs/c1s1_123456.jpg"
cv2.imwrite(save_path, im0)


