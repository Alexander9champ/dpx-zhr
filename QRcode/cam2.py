import cv2
from datetime import datetime

# 初始化摄像头
cap = cv2.VideoCapture(2)

# 设置摄像头分辨率为640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 初始化视频录制标志
recording = False
out = None

while True:
    # 捕获视频帧
    ret, frame = cap.read()
    if not ret:
        break

    # 显示当前视频帧
    cv2.imshow("Camera", frame)

    # 检查键盘输入
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 "s" 键，开始录制
    if key == ord("s") and not recording:
        # 生成当前时间的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.mp4"
        
        # 定义视频编码器（H.264），输出文件名和帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用H.264编码
        out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
        recording = True
        print(f"Recording started: {filename}")

    # 如果按下 "k" 键，停止录制并保存
    elif key == ord("k") and recording:
        recording = False
        out.release()
        print("Recording stopped and saved.")

    # 如果按下 "q" 键，退出
    elif key == ord("q"):
        break

    # 如果正在录制，写入当前帧到文件
    if recording:
        out.write(frame)

# 释放摄像头和关闭窗口
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
