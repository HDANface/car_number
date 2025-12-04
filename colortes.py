import cv2  
import numpy as np  
import os


'''
HSV颜色筛选程序:
在path中读取图像,然后在弹出窗口中显示图像,并在窗口中添加6个滑动条,用于调整HSV值,并在窗口中添加一个按钮,用于确定当前的HSV值范围.
'''

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'dataset', '106_川AT081D.jpg')  
def empty(a):  
    try:
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")  
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")  
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")  
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")  
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")  
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")  
        return h_min, h_max, s_min, s_max, v_min, v_max  
    except Exception as e:

        return 0, 179, 0, 255, 0, 255

# 全局变量存储HSV值
global_hsv_values = [0, 179, 0, 255, 0, 255]

def mouse_callback(event, x, y, flags, param):
    global global_hsv_values
    if event == cv2.EVENT_LBUTTONDOWN:
        if 100 <= x <= 200 and 50 <= y <= 100:
            # 打印当前HSV值
            print("\n=== 确定按钮点击 ===")
            print(f"当前HSV值: H_min={global_hsv_values[0]}, H_max={global_hsv_values[1]}")
            print(f"S_min={global_hsv_values[2]}, S_max={global_hsv_values[3]}")
            print(f"V_min={global_hsv_values[4]}, V_max={global_hsv_values[5]}")
            print(f"HSV范围: {[global_hsv_values[0], global_hsv_values[2], global_hsv_values[4]]} 到 {[global_hsv_values[1], global_hsv_values[3], global_hsv_values[5]]}")
            print("================\n")
            # 安全关闭应用程序
            cv2.destroyAllWindows()
            exit(0)
    
# 创建一个窗口，放置6个滑动条  
cv2.namedWindow("TrackBars")  
cv2.resizeWindow("TrackBars", 640, 240)

# 创建按钮窗口
cv2.namedWindow("Controls")
cv2.resizeWindow("Controls", 300, 150)

# 设置鼠标事件回调
cv2.setMouseCallback("Controls", mouse_callback)  

# 设置更合理的默认HSV参数，避免初始全黑
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)  
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)  
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)  
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)  
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)  
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)  
  
while True:  
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"无法读取图片: {path}")
        img = np.zeros((400, 600, 3), dtype=np.uint8)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  

    h_min, h_max, s_min, s_max, v_min, v_max = empty(0)  
    lower = np.array([h_min, s_min, v_min])  
    upper = np.array([h_max, s_max, v_max])  

    mask = cv2.inRange(imgHSV, lower, upper)  
    # 对原图图像进行按位与的操作，掩码区域保留  
    imgResult = cv2.bitwise_and(img, img, mask=mask)  
    
    display_width = 800
    display_height = 600
    
    #
    img_resized = cv2.resize(img, (display_width, display_height))
    mask_resized = cv2.resize(mask, (display_width, display_height))
    imgResult_resized = cv2.resize(imgResult, (display_width, display_height))
    
    # 更新全局HSV值
    global_hsv_values = [h_min, h_max, s_min, s_max, v_min, v_max]
    

    cv2.imshow("Original", img_resized)
    cv2.imshow("Mask", mask_resized)  
    cv2.imshow("Result", imgResult_resized)  
    
    button_window = np.ones((150, 300, 3), dtype=np.uint8) * 200  

    cv2.rectangle(button_window, (100, 50), (200, 100), (0, 150, 0), -1)  

    cv2.putText(button_window, "OVER", (125, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Controls", button_window)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:  
        break


cv2.destroyAllWindows()