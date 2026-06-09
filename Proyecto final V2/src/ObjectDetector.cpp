#include "ObjectDetector.hpp"
#include <fstream>
#include <iostream>

bool ObjectDetector::load(const std::string& modelPath,
                           const std::string& classesPath,
                           float conf, float nms) {
    confThresh_ = conf;
    nmsThresh_  = nms;

    // Load classes
    std::ifstream cf(classesPath);
    if (cf.is_open()) {
        std::string line;
        while (std::getline(cf, line))
            if (!line.empty()) classes_.push_back(line);
    }

    // Try to load model (optional feature — silent fail)
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);
    } catch (...) {
        std::cout << "[Detector] Model not found: " << modelPath
                  << " — detection disabled.\n";
        return false;
    }

    if (net_.empty()) {
        std::cout << "[Detector] Failed to load model — detection disabled.\n";
        return false;
    }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Detect YOLOv8 by checking output layer count / shape
    // YOLOv8 exports as [1, 4+nc, 8400], v5 as [1, 25200, 5+nc]
    auto outNames = net_.getUnconnectedOutLayersNames();
    isYolov8_ = (outNames.size() == 1);  // heuristic; works for standard exports

    loaded_ = true;
    std::cout << "[Detector] Loaded " << modelPath
              << (isYolov8_ ? " (YOLOv8)" : " (YOLOv5)") << "\n";
    return true;
}

std::vector<Detection> ObjectDetector::detect(const cv::Mat& bgr, float confOverride) {
    if (!loaded_ || bgr.empty()) return {};

    if (confOverride >= 0.0f) confThresh_ = confOverride;

    cv::Mat blob;
    cv::dnn::blobFromImage(bgr, blob, 1.0 / 255.0,
                           cv::Size(inputW_, inputH_),
                           cv::Scalar(), true, false);
    net_.setInput(blob);

    std::vector<cv::Mat> outs;
    net_.forward(outs, net_.getUnconnectedOutLayersNames());
    if (outs.empty()) return {};

    const cv::Mat& out = outs[0];
    if (isYolov8_)
        return postprocessV8(out, bgr.cols, bgr.rows);
    else
        return postprocessV5(out, bgr.cols, bgr.rows);
}

// YOLOv5: output [1, num_det, 5+nc]
std::vector<Detection> ObjectDetector::postprocessV5(const cv::Mat& raw,
                                                       int origW, int origH) {
    const float* data = reinterpret_cast<const float*>(raw.data);
    int rows = raw.size[1];
    int dims = raw.size[2];
    int nc   = dims - 5;
    if (nc <= 0) return {};

    float sx = static_cast<float>(origW)  / inputW_;
    float sy = static_cast<float>(origH)  / inputH_;

    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       ids;

    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * dims;
        float objConf    = row[4];
        if (objConf < confThresh_) continue;

        // Best class
        int   bestCls  = 0;
        float bestConf = 0;
        for (int c = 0; c < nc; ++c) {
            float cf = row[5 + c] * objConf;
            if (cf > bestConf) { bestConf = cf; bestCls = c; }
        }
        if (bestConf < confThresh_) continue;

        float cx = row[0] * sx, cy = row[1] * sy;
        float w  = row[2] * sx, h  = row[3] * sy;
        boxes.push_back({(int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h});
        scores.push_back(bestConf);
        ids.push_back(bestCls);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, confThresh_, nmsThresh_, idx);

    std::vector<Detection> out;
    for (int i : idx) {
        Detection d;
        d.box        = boxes[i] & cv::Rect(0, 0, origW, origH);
        d.confidence = scores[i];
        d.classId    = ids[i];
        d.label      = (ids[i] < (int)classes_.size()) ? classes_[ids[i]]
                                                        : std::to_string(ids[i]);
        out.push_back(d);
    }
    return out;
}

// YOLOv8: output [1, 4+nc, 8400]  (transposed vs v5)
std::vector<Detection> ObjectDetector::postprocessV8(const cv::Mat& raw,
                                                       int origW, int origH) {
    // Reshape to [4+nc, 8400] then transpose to [8400, 4+nc]
    cv::Mat t = raw.reshape(1, raw.size[1]);  // [4+nc, 8400]
    cv::Mat data;
    cv::transpose(t, data);                   // [8400, 4+nc]

    int rows = data.rows;
    int nc   = data.cols - 4;
    if (nc <= 0) return {};

    float sx = static_cast<float>(origW)  / inputW_;
    float sy = static_cast<float>(origH)  / inputH_;

    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       ids;

    for (int i = 0; i < rows; ++i) {
        const float* row = data.ptr<float>(i);
        int   bestCls  = 0;
        float bestConf = 0;
        for (int c = 0; c < nc; ++c) {
            if (row[4 + c] > bestConf) { bestConf = row[4 + c]; bestCls = c; }
        }
        if (bestConf < confThresh_) continue;

        float cx = row[0] * sx, cy = row[1] * sy;
        float w  = row[2] * sx, h  = row[3] * sy;
        boxes.push_back({(int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h});
        scores.push_back(bestConf);
        ids.push_back(bestCls);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, confThresh_, nmsThresh_, idx);

    std::vector<Detection> out;
    for (int i : idx) {
        Detection d;
        d.box        = boxes[i] & cv::Rect(0, 0, origW, origH);
        d.confidence = scores[i];
        d.classId    = ids[i];
        d.label      = (ids[i] < (int)classes_.size()) ? classes_[ids[i]]
                                                        : std::to_string(ids[i]);
        out.push_back(d);
    }
    return out;
}
