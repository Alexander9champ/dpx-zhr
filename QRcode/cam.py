import cv2

# 初始化摄像头
cap = cv2.VideoCapture(2)

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
        # 定义视频编码器，输出文件名和帧率
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("output.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        recording = True
        print("Recording started...")

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
