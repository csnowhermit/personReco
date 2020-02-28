
import os
import cv2
import threading

'''
    自定义帧缓冲区：每次读取拿最新的帧，避免卡顿。
    cv2自带的缓冲区的缺陷：read每帧数据到缓冲区，读取按序读取。但处理的速度远赶不上读取的速度，会造成帧挤压越来越多，看起来越来越卡。
    opencv能读取当前帧（会在一定时间清空帧缓冲区），但这个时机我们无法控制，没有api让我们手动清空帧缓冲区。
    视频30+fps，我处理能力（2g显存）每帧0.41+秒。
'''
class Stack:
    def __init__(self, stack_size):
        self.items = []
        self.stack_size = stack_size

    def is_empty(self):
        return len(self.items) == 0

    def pop(self):
        return self.items.pop()

    def peek(self):
        if not self.isEmpty():
            return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def push(self, item):
        if self.size() >= self.stack_size:
            for i in range(self.size() - self.stack_size + 1):
                self.items.remove(self.items[0])
        self.items.append(item)

'''
    开启视频流，开启线程
'''
def capture_thread(video_path, frame_buffer, lock):
    print("capture_thread start")
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    while True:
        return_value, frame = vid.read()
        if return_value is not True:
            break
        lock.acquire()
        frame_buffer.push(frame)
        lock.release()
        cv2.waitKey(25)

'''
    播放视频流，播放线程
'''
def play_thread(frame_buffer, lock):
    print("detect_thread start")
    print("detect_thread frame_buffer size is", frame_buffer.size())

    while True:
        if frame_buffer.size() > 0:
            lock.acquire()
            frame = frame_buffer.pop()
            lock.release()
            # 每次拿最新的，显示
            cv2.imshow("result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    path = '../testData/input/120_150_2.mp4'
    rtsp_url = ''
    frame_buffer = Stack(3)    # 3帧3帧地过
    lock = threading.RLock()
    t1 = threading.Thread(target=capture_thread, args=(path, frame_buffer, lock))
    t1.start()
    t2 = threading.Thread(target=play_thread, args=(frame_buffer, lock))
    t2.start()
