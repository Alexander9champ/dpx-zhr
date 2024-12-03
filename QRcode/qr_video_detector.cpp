#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

int main() {
    // 打开视频文件或摄像头
    std::string videoPath = "20241026_225827.mp4"; // 如果要从文件读取，替换为视频文件路径
    cv::VideoCapture cap(videoPath); // 使用摄像头时替换为 cv::VideoCapture(0)

    if (!cap.isOpened()) {
        std::cerr << "无法打开视频!" << std::endl;
        return -1;
    }

    // 创建二维码检测器
    cv::QRCodeDetector qrDecoder;

    cv::Mat frame;
    while (true) {
        cap >> frame; // 读取视频帧

        if (frame.empty()) {
            std::cerr << "视频读取结束!" << std::endl;
            break;
        }

        // 预处理：转灰度 + 直方图均衡化
        cv::Mat gray, enhanced;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // 转为灰度图
        cv::equalizeHist(gray, enhanced);             // 直方图均衡化

        // 检测并解码二维码
        std::string decodedText;
        std::vector<cv::Point> points;
        decodedText = qrDecoder.detectAndDecode(enhanced, points);

        if (!decodedText.empty()) {
            std::cout << "检测到二维码内容: " << decodedText << std::endl;

            // 绘制二维码位置框
            if (points.size() == 4) {
                for (int i = 0; i < 4; i++) {
                    cv::line(frame, points[i], points[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
                }
            }
        } else {
            std::cout << "未检测到二维码或解码失败" << std::endl;

            // 调试：处理检测到的二维码框
            if (!points.empty()) {
                std::cout << "检测到二维码位置，但解码失败，位置点如下:" << std::endl;
                for (size_t i = 0; i < points.size(); i++) {
                    std::cout << "点 " << i + 1 << ": " << points[i] << std::endl;
                }

                // 检查点的合法性
                bool validPoints = true;
                for (const auto& point : points) {
                    if (point.x < 0 || point.y < 0 || point.x >= frame.cols || point.y >= frame.rows) {
                        validPoints = false;
                        break;
                    }
                }

                if (validPoints) {
                    // 提取二维码区域
                    cv::Rect boundingBox = cv::boundingRect(points);
                    cv::Mat qrCodeRegion = frame(boundingBox);

                    // 放大二维码区域
                    cv::Mat enlargedQRCode;
                    cv::resize(qrCodeRegion, enlargedQRCode, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);

                    // 二值化处理
                    cv::Mat binaryQRCode;
                    cv::cvtColor(enlargedQRCode, binaryQRCode, cv::COLOR_BGR2GRAY); // 转为灰度图
                    cv::threshold(binaryQRCode, binaryQRCode, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

                    // 再次尝试解码
                    std::string decodedTextEnhanced = qrDecoder.detectAndDecode(binaryQRCode);
                    if (!decodedTextEnhanced.empty()) {
                        std::cout << "放大并二值化后成功解码二维码内容: " << decodedTextEnhanced << std::endl;
                    } else {
                        std::cout << "放大并二值化后仍然无法解码二维码" << std::endl;
                    }

                    // 显示裁剪和处理后的二维码区域
                    cv::imshow("裁剪的二维码区域", qrCodeRegion);
                    cv::imshow("放大的二维码区域", enlargedQRCode);
                    cv::imshow("二值化后的二维码区域", binaryQRCode);
                } else {
                    std::cout << "检测到的二维码点无效，无法裁剪或处理二维码区域。" << std::endl;
                }
            } else {
                std::cout << "未检测到二维码位置。" << std::endl;
            }
        }

        // 显示当前帧
        cv::imshow("二维码识别", frame);

        // 如果用户按下 'q' 键，则退出
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
