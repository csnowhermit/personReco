import cv2

global frame
global point1, point2


# 截取需要查找的行人图片
def on_mouse(event, x, y, flags, param):
	global frame, point1, point2
	img2 = frame.copy()
	if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
		point1 = (x, y)
		cv2.circle(img2, point1, 10, (0, 255, 0), 5)
		cv2.imshow('image', img2)
	elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
		cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
		cv2.imshow('image', img2)
	elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
		point2 = (x, y)
		cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
		cv2.imshow('image', img2)
		min_x = min(point1[0], point2[0])
		min_y = min(point1[1], point2[1])
		width = abs(point1[0] - point2[0])
		height = abs(point1[1] - point2[1])
		cut_img = frame[min_y:min_y + height, min_x:min_x + width]
		print("cut_img", type(cut_img), cut_img.shape)
		save_file = 'query/0001_c1s1_0_%s.jpg' % min_x
		cv2.imwrite(save_file, cut_img)
		cv2.resize(save_file, save_file[0:save_file.index(".")] + "_00" + save_file[save_file.index("."):], )

if __name__ == '__main__':

	videopath = "D:/workspace/Pycharm_Projects/mx_AI/person_search/data/samples/台湾主播抱怨广州南站“换钱不方便”.mp4"
	videoCapture = cv2.VideoCapture(videopath)
	# Read image
	success, frame = videoCapture.read()
	while success:
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', on_mouse)
		cv2.imshow('image', frame)
		cv2.waitKey(0)
		success, frame = videoCapture.read()
