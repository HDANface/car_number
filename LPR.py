import cv2
import numpy as np


# 图像处理
class PhotoProcess:

    def __init__(self, image):
        self.img = cv2.imread(image)

    # 图片查看
    def ShowImage(self):
        cv2.imshow("image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 对车牌进行定位,
    def Positioning(self):
        #边缘定位(canny算法):
        self.img = cv2.Canny(self.img, 100, 500)
        #生成方框
        contour = cv2.findContours(self.img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 对方框进行筛选(这使用Area面积筛选)
        sorted_contour = sorted(contour, key=cv2.contourArea, reverse=True)[:10]
        cv2.drawContours(self.img, sorted_contour, -1, (0, 255, 0), 2)

    # 图片处理转为灰度图
    def ChangeColor(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.bilateralFilter(self.img, 13, 17, 17)


car = PhotoProcess('./dataset/R (1).png')
car.ChangeColor()
car.Positioning()
car.ShowImage()