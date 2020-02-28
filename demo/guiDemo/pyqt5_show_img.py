from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2

img=cv2.imread('../../testData/input/c1s1_123456.jpg')

app = QApplication(sys.argv)
window = QWidget()
label=QLabel(window)

img=cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
showImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
label.setPixmap(QPixmap.fromImage(showImage))
window.show()
sys.exit(app.exec_())