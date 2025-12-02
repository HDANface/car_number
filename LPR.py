import cv2
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 图像处理
class PhotoProcess:

    def __init__(self, image):
        try:
            with open(image, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                self.img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except:
            logging.error(f"无法读取图像文件: {image}")
            self.img = None


    # 图片查看
    def ShowImage(self):
        self.img = cv2.resize(self.img, (1024, 800))
        cv2.imshow("image", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _check_roi_color(self, roi):
        """检查ROI区域的颜色"""
        # 转换为HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 蓝色范围
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 计算蓝色像素比例
        blue_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])

        # 如果蓝色像素超过20%，认为是蓝色车牌
        if blue_ratio > 0.2:
            return 'blue'
        return 'unknown'

    # 对车牌进行定位,
    def Positioning(self):

        # 膨胀边缘用于处理车牌和号码分开问题
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(self.img, kernel, iterations=1)
        #边缘定位(canny算法):
        edges = cv2.Canny(dilated, 30, 150)

        #生成方框
        contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 对方框进行筛选(这使用Area面积筛选)
        sorted_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # 只框选蓝色区域
        for car_card in sorted_contour:
            # 筛选闭合图形
            peri = cv2.arcLength(car_card, True)
            # 查找长方形
            approx = cv2.approxPolyDP(car_card, 0.018*peri, True)
            if len(approx) == 4:
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(car_card)
                # 截取ROI区域进行颜色判断
                roi = self.img[y:y+h, x:x+w]
                if roi.size > 0:
                    color = self._check_roi_color(roi)  # 检查ROI颜色
                    if color == 'blue':
                        cv2.drawContours(self.img, [approx], -1, (255, 0, 0), 5)  # 蓝色框
                        logging.info("找到蓝色车牌区域")
                        return x, y, w, h  # 返回车牌位置

        logging.info("未找到蓝色车牌")
        return None

    # 图片裁剪
    def CutImg(self, plate_position):
        if plate_position is None:
            logging.error("没有传入有效的车牌位置")
            return None

        x, y, w, h = plate_position

        # 确保坐标在图像范围内
        height, width = self.img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            logging.error("裁剪区域无效")
            return None

        # 裁剪图像
        cropped_img = self.img[y:y+h, x:x+w]

        # 保存裁剪结果
        cv2.imwrite('cropped_plate.jpg', cropped_img)
        logging.info(f"车牌已裁剪，保存为 cropped_plate.jpg，尺寸: {w}x{h}")

        return cropped_img




    # 图片处理转为灰度图
    def ChangeColor(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.bilateralFilter(self.img, 13, 17, 17)

    # 调试函数
    def __abs__(self):
        pass


def main():
    car = PhotoProcess('dataset/091606539239981.jpg')

    plate_position = car.Positioning()  # 获取车牌位置

    if plate_position:
        logging.info(f"车牌位置: {plate_position}")

        # 裁剪车牌
        cropped_plate = car.CutImg(plate_position)

        # 显示原图（带框选）
        car.ShowImage()

        # 显示裁剪的车牌
        if cropped_plate is not None:
            # 显示裁剪结果
            cv2.imshow("裁剪的车牌", cropped_plate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

