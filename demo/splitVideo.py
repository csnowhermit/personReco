import cv2

'''
    拆分视频成图片
'''

vidcap = cv2.VideoCapture('D:/logs/演示版/120_150_2-忽略隔壁车厢.mp4')    # 20200108080005.2.6577.0.33.10.0.5-2.mp4

# vidcap = cv2.VideoCapture("../testData/output鱼眼相机/20200224_100418.mp4")
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    if success:
        cv2.imwrite("D:/logs/frame_%06d.jpg" % count, image)  # save frame as JPEG file
        count += 1
    if count % 1000 == 0:
        print("已保存 %d 张" % count)

print("已保存 %d 张" % count)