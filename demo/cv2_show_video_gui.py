import cv2

video_path = "../testData/input/120_150_2.mp4"

vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    if success:
        cv2.imshow("video_show", image)
        cv2.waitKey(10)

print("finished")