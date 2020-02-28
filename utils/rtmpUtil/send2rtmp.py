import cv2
import subprocess

'''
    获取摄像头视频到rtmp，推流
'''

rtsp = "rtsp_address"
rtmp = "rtmp://localhost:1935/live/home"

# cap = cv2.VideoCapture("C:/Users/ASUS/Desktop/新建文件夹/120_150_2-只标中文名称.mp4")
cap = cv2.VideoCapture(0)
# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size = (1280, 720)
sizeStr = str(size[0]) + 'x' + str(size[1])

command = ['ffmpeg',
           '-y', '-an',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', sizeStr,
           '-r', '25',
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp]

pipe = subprocess.Popen(command
                        , shell=False
                        , stdin=subprocess.PIPE
                        )

while 1:
    ret, frame = cap.read()
    # print("frame:", type(frame), frame.shape)
    # 这里窗口和页面只能开一个
    # cv2.imshow("frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    pipe.stdin.write(frame.tostring())

cap.release()
pipe.terminate()
