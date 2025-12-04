import cv2
import numpy as np
import logging
import csv
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# 图像处理
class PhotoProcess:

    def __init__(self, image_path):
        self.image_path = image_path
        self.success_read = False
        self.success_cut = False
        self.cropped_path = None
        
        try:
            with open(image_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                self.img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.success_read = True
        except Exception as e:
            logging.error(f"无法读取图像文件 {image_path}: {e}")
            self.img = None


    def ShowImage(self):
        """图片查看"""
        show_img = cv2.resize(self.img, (1024, 800))
        cv2.imshow("image", show_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # 对车牌进行定位
    def Positioning(self):
        """定位筛选"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        avg_v = np.mean(hsv[:,:,2])

        (lower_blue, upper_blue) = (
        (np.array([111, 233, 97]), np.array([136, 255, 255])) 
        if avg_v < 80 
        else (np.array([110, 128, 52]), np.array([125, 255, 77]))
        )

        # 创建遮罩
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # cv2.imshow("mask", mask)


        # 闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选轮廓
        for c in contours:
            if cv2.contourArea(c) < 1100:
                continue

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)

            box = np.int64(box)

            width = rect[1][0]
            height = rect[1][1]

            if width < height:
                width, height = height, width

            if height == 0: continue

            ratio = width / height

            if 2.0 < ratio < 5.5:
                x, y, w, h = cv2.boundingRect(c)

                # 车牌标记
                cv2.drawContours(self.img, [box], 0, (0, 255, 0), 2)
                logging.info(f"找到蓝色车牌区域 (HSV法): {x},{y},{w},{h}")
                return x, y, w, h

        logging.info("未找到蓝色车牌")
        return None

    # 图片裁剪
    def CutImg(self, plate_position):
        if plate_position is None:
            logging.error("没有传入有效的车牌位置")
            return None

        # 解包
        x, y, w, h = plate_position

        # 四角计算
        height, width = self.img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            logging.error("裁剪区域无效")
            return None

        cropped_img = self.img[y:y+h, x:x+w]

        # 生成唯一的裁剪后文件名并保存到exportdata文件夹
        base_name = os.path.basename(self.image_path)
        name, ext = os.path.splitext(base_name)
        self.cropped_path = os.path.join('exportdata', f'cropped_{name}.jpg')
        
        # 写入文件 - 使用绝对路径确保中文路径正确
        absolute_path = os.path.abspath(self.cropped_path)
        cv2.imencode('.jpg', cropped_img)[1].tofile(absolute_path)  
        self.success_cut = True
        logging.info(f"车牌已裁剪，保存为 {self.cropped_path}，尺寸: {w}x{h}")
        return cropped_img

    @staticmethod
    def ExportCSV(data_list, filename='plate_positions.csv'):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(['序号', '原图', '是否成功读取', '成功截取', '截取后图片'])
            for i, data in enumerate(data_list, 1):
                writer.writerow([i, data['image_path'], data['success_read'], data['success_cut'], data['cropped_path']])


def main():
    # 获取dataset文件夹中的所有图像文件
    dataset_path = 'CLPD_1200'
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 处理所有图像并收集数据
    results = []
    for image_path in image_files:
        logging.info(f"\n处理图像: {image_path}")
        car = PhotoProcess(image_path)
        
        if car.success_read:
            plate_position = car.Positioning()
            if plate_position:
                logging.info(f"车牌位置: {plate_position}")
                car.CutImg(plate_position)
        
        # 收集处理结果
        results.append({
            'image_path': image_path,
            'success_read': '是' if car.success_read else '否',
            'success_cut': '是' if car.success_cut else '否',
            'cropped_path': car.cropped_path or ''
        })
    
    # 导出CSV文件
    PhotoProcess.ExportCSV(results)
    logging.info(f"\nCSV文件导出完成，共处理 {len(results)} 个图像文件")

if __name__ == '__main__':
    main()