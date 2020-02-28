import time
from gui.GUI.widgets import *
from utils.channel import Channel

sub = Channel()

class Extractor_GUI():
    def __init__(self, width, height):
        self.__init_gui(width, height)
        # self.__init_model()

    def __init_gui(self, width, height):
        self.width = width
        self.height = height

        self.window = tk.Tk()
        self.window.wm_title('识人模型演示')
        self.window.config(background = '#FFFFFF')
        # self.window.geometry("960x640")    # 窗体大小
        self.window.geometry("1077x640")    # 窗体大小

        self.canvas = ICanvas(self.window, width = width, height = height)
        self.canvas.grid(row = 0, column = 0)

        self.fm_control = tk.Frame(self.window, width=800, height=100, background = '#FFFFFF')
        self.fm_control.grid(row = 1, column=0, padx=10, pady=2)
        self.btn_prev_frame = tk.Button(self.fm_control, text='开始', command = self.__action_read_frame)
        self.btn_prev_frame.grid(row = 0, column=0, padx=10, pady=2)
        self.lb_current_frame = tk.Label(self.fm_control, background = '#FFFFFF')
        self.lb_current_frame.grid(row = 0, column=1, padx=10, pady=2)
        self.lb_current_frame['text'] = '----'
        self.btn_next_frame = tk.Button(self.fm_control, text='结束', command = self.__action_stop)
        self.btn_next_frame.grid(row = 0, column=2, padx=10, pady=2)

        # 窗体右侧部分
        self.fm_status = tk.Frame(self.window, width = 200, height = 800, background = '#FFFFFF')
        self.fm_status.grid(row = 0, column=1, padx=10, pady=2)

        # label标签，显示各种人数
        self.info_label = tk.Label(self.fm_status, background='#FFFFFF', font=("楷体", 12),
                                   justify="left", foreground="red", anchor="center")    # foreground，字体颜色
        self.info_label.grid(row = 0, column=1, padx=10, pady=2)
        self.info_label['text'] = '点击下方“开始”按钮演示程序'


    def __action_read_frame(self):
        self.from_video()

    '''
        画布清屏，提示语初始化
    '''
    def __action_stop(self):
        self.canvas.destroy()
        self.canvas = ICanvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)

        self.info_label['text'] = '点击下方“开始”按钮演示程序'
        self.launch()

    def from_video(self):
        cap = cv2.VideoCapture("D:/logs/演示版/识别填乘.mp4")
        fo = open("D:/logs/演示版/识别填乘.txt", 'r', encoding="utf-8")
        # cap = cv2.VideoCapture("rtmp://localhost:1935/live/home")

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        fps = 0
        if int(major_ver) < 3:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS:", fps)
        # sleep_time = float(format(float(1)/float(fps),'.2f'))
        # print("sleep_time:", sleep_time)
        idx = 0
        while True:
            # time.sleep(sleep_time)    # 参数为秒
            # 先判断文字，避免多读一帧图像造成的error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'错误
            line = fo.readline().strip("\n")
            if len(line) == 0:  # 文件内容完了，就退出循环
                break
            # print("line:", line)
            self.info_label['text'] = line.replace("{", "").replace("}", "").replace("'", "").replace(", ", "\n").replace(",", "\n")

            ret,img = cap.read()
            # print("idx:", idx)
            # idx = idx + 1
            # print(frame)
            # img = cv2.transpose(img)
            # img = cv2.flip(img,1)
            # print(img.shape)

            # print("img:", type(img))
            self.canvas.add(img, self.width, self.height)
            # self.text_person_frame.tag_add()
            self.window.update_idletasks()
            self.window.update()
        # 演示完成后
        self.__action_stop()

    def launch(self):
        self.window.mainloop()

def main():
    width, height = 800, 600
    # width, height = 1280, 720
    ext = Extractor_GUI(width, height)
    ext.launch()
    # ext.from_video()

if __name__ == '__main__':
    main()
