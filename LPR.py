import cv2
import numpy as np
import logging
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class PhotoPositioning:
    def __init__(self, image_path):
        self.img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.img is None:
            logging.error(f"无法读取图像: {image_path}")

    def ShowImage(self):
        if self.img is not None:
            show_img = cv2.resize(self.img, (1024, 800))
            cv2.imshow("Result", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # --- 新增方法：纹理对比度校验 ---
    def verify_texture(self, img_region):
        """检查区域内部是否有足够的纹理对比度（排除纯色地面/墙壁）"""
        if img_region is None or img_region.size == 0: return False
        try:
            # 转为灰度图
            gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
            # 计算灰度方差。车牌内部有文字，方差应该较高。
            # 地面、墙壁或绿化带通常比较平坦，方差较低。
            variance = np.var(gray)
            # 阈值经验值。低于此值通常认为是背景干扰。
            # 可以根据实际效果微调 (例如在 800 - 1500 之间尝试)
            return variance > 1000  
        except Exception:
            return False

    # --- 修改：增加颜色参数，区分长宽比 ---
    def verify_scale(self, rotate_rect, color_name):
        (x, y), (width, height), angle = rotate_rect
        if width == 0 or height == 0: return False
        
        # 矫正长宽，确保 width 是长边
        if height > width:
            width, height = height, width
            angle = angle + 90

        aspect_ratio = width / height
        
        # 根据颜色设定不同的比例范围
        if color_name == "yellow":
            # 黄牌兼容单层和双层，范围较宽
            min_ratio, max_ratio = 1.5, 5.0
        else:
            # 蓝牌和绿牌通常是单层，收紧下限以排除干扰
            min_ratio, max_ratio = 2.2, 5.5

        if min_ratio < aspect_ratio < max_ratio:
             area = width * height
             # 稍微放宽最小面积，避免漏检远处车牌
             if 800 < area < 50000:
                 return True
        return False

    def Positioning(self):
        if self.img is None: return None
        
        # 使用高斯模糊降噪
        blur = cv2.GaussianBlur(self.img, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        img_h, img_w = self.img.shape[:2]

        # --- 核心修改1: 优化多色 HSV 字典 ---
        color_configs = {
            # 蓝牌：微调以适应夜间（降低饱和度下限），同时保持对白天的敏感度
            "blue":   (np.array([100, 60, 60]), np.array([130, 255, 255])),
            # 黄牌：显著提高饱和度和亮度下限，排除暗淡的地面和减速带
            "yellow": (np.array([15, 120, 120]), np.array([35, 255, 255])),
            # 绿牌：收紧色相范围，提高饱和度下限，排除自然植物
            "green":  (np.array([50, 80, 80]), np.array([90, 255, 255])),
        }

        candidates = [] 

        for color_name, (lower, upper) in color_configs.items():
            mask = cv2.inRange(hsv, lower, upper)
            
            # 形态学操作
            kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_x)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_x)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                rect = cv2.minAreaRect(c)
                # 传入颜色参数进行比例校验
                if self.verify_scale(rect, color_name):
                    box = np.int64(cv2.boxPoints(rect))
                    
                    # --- 核心修改2: 新增纹理校验 (关键步骤) ---
                    # 1. 获取候选区域的正矩形包围盒用于裁剪
                    rx, ry, rw, rh = cv2.boundingRect(box)
                    # 2. 边界安全检查，防止越界
                    rx, ry = max(0, rx), max(0, ry)
                    rw, rh = min(rw, img_w - rx), min(rh, img_h - ry)
                    
                    if rw > 0 and rh > 0:
                        candidate_img = self.img[ry:ry+rh, rx:rx+rw]
                        # 3. 如果纹理不够丰富（方差低），则认为是误报，跳过
                        if not self.verify_texture(candidate_img):
                            continue
                    else:
                        continue
                    # ----------------------------------------
                    
                    # 基础过滤逻辑
                    rect_w, rect_h = rect[1]
                    if rect_w < rect_h: rect_w, rect_h = rect_h, rect_w
                    
                    # 位置和大小过滤
                    if rect[0][1] < img_h * 0.2: continue
                    if rect_w > img_w * 0.85: continue
                    
                    # 饱满度过滤
                    contour_area = cv2.contourArea(c)
                    rect_area = rect_w * rect_h
                    if rect_area == 0: continue
                    solidity = contour_area / rect_area
                    
                    # 适当提高饱满度要求
                    if solidity > 0.50:
                        # 评分逻辑
                        ratio = rect_w / rect_h
                        # 根据颜色设定标准比例
                        std_ratio = 3.14
                        if color_name == "yellow" and ratio < 2.5:
                             std_ratio = 2.0 # 双层黄牌
                        
                        weight = 1 - (abs(ratio - std_ratio) * 0.5)
                        score = rect_area * weight
                        
                        candidates.append((score, rect, box, color_name))

        # 全局择优
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_rect, best_box, best_color = candidates[0]
            
            # 绘图
            draw_color = (0, 0, 255)
            if best_color == "green": draw_color = (0, 255, 0)
            elif best_color == "yellow": draw_color = (0, 255, 255)
            
            cv2.drawContours(self.img, [best_box], 0, draw_color, 2)
            
            x, y, w, h = cv2.boundingRect(best_box)
            logging.info(f"成功定位 [{best_color}] 车牌: {x},{y},{w},{h}, score: {best_score:.1f}")
            return x, y, w, h, best_color

        logging.info("未定位到车牌")
        return None, None

    # CutImg 方法保持不变
    def CutImg(self, plate_info):
        if plate_info[0] is None: return None
        x, y, w, h = plate_info[0], plate_info[1], plate_info[2], plate_info[3]
        h_img, w_img = self.img.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_img - x), min(h, h_img - y)
        return self.img[y:y+h, x:x+w]

# 主程序部分不需要修改，直接运行即可。
if __name__ == '__main__':

    input_dir = 'CLPD_1200'
    output_dir = 'exportdate_02'


    total_images = 0
    successfully_recognized = 0

    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    total_images = len(image_files)

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            positioning = PhotoPositioning(image_path)

            # 接收两个返回值：坐标信息 和 颜色
            # *pos 解包前4个坐标参数, color_type 接收颜色字符串
            result = positioning.Positioning()
            x, y, w, h, color_type = result

            if x is not None:
                # 传入坐标信息进行裁剪
                cut_img = positioning.CutImg((x, y, w, h))
                if cut_img is not None:
                    # 可以在文件名中加入颜色标识，方便你查看分类效果
                    # 例如: "100_川A_blue.jpg"
                    file_name_no_ext = os.path.splitext(image_file)[0]
                    new_filename = f"{file_name_no_ext}_{color_type}.jpg"

                    output_path = os.path.join(output_dir, new_filename)
                    cv2.imencode('.jpg', cut_img)[1].tofile(output_path)
                    successfully_recognized += 1
        except Exception as e:
            logging.error(f"处理图片 {image_file} 时出错: {str(e)}")

        # 输出统计结果

print("=== 处理结果统计 ===")

print(f"成功导入exportdate文件夹的图片总张数: {total_images}")

print(f"成功识别的图片张数: {successfully_recognized}")

if total_images > 0:

    recognition_rate = (successfully_recognized / total_images) * 100

    print(f"识别成功率: {recognition_rate:.2f}%")

else:

    print("识别成功率: 0.00%")

    print("====================")