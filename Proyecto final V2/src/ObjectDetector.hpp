#pragma once
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Detection {
    cv::Rect  box;
    int       classId;
    float     confidence;
    std::string label;
    float     depthMm = 0.0f;  // filled by caller from disparity map
};

class ObjectDetector {
public:
    // modelPath: ONNX file. classesPath: text file, one label per line.
    // Returns false (and stays silent) if model file not found — detection is optional.
    bool load(const std::string& modelPath,
              const std::string& classesPath,
              float confThreshold = 0.45f,
              float nmsThreshold  = 0.45f);

    bool isLoaded() const { return loaded_; }

    // Run inference on BGR frame. Returns detections in original frame coordinates.
    std::vector<Detection> detect(const cv::Mat& bgr, float confOverride = -1.0f);

    void setConfThreshold(float t) { confThresh_ = t; }
    void setNmsThreshold (float t) { nmsThresh_  = t; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> classes_;
    float confThresh_ = 0.45f;
    float nmsThresh_  = 0.45f;
    bool  loaded_     = false;
    bool  isYolov8_   = false;  // v8 has transposed output vs v5
    int   inputW_     = 640;
    int   inputH_     = 640;

    std::vector<Detection> postprocessV5(const cv::Mat& out,
                                          int origW, int origH);
    std::vector<Detection> postprocessV8(const cv::Mat& out,
                                          int origW, int origH);
};
