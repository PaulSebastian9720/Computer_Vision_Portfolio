#pragma once
#include <array>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class HandTracker {
public:
  // Landmark pixel coords in the ORIGINAL image frame
  struct Lm {
    float x, y;
  };

  struct Hand {
    std::array<Lm, 21> lm{}; // MediaPipe 21-point skeleton
    bool isRight = false;
    float conf = 0.0f;
    float radiusPx = 0.0f; // palm radius in pixels (proxy for depth)
    cv::Point palmCenter{0, 0};
  };

  // palmOnnx   : models/palm_detection.onnx  (128×128 input, 896 anchors)
  // lmOnnx     : models/hand_landmark.onnx   (224×224 input)
  bool load(const std::string &palmOnnx, const std::string &lmOnnx,
            float detThresh = 0.60f, float nmsThresh = 0.40f);

  bool isLoaded() const { return loaded_; }

  // Detect up to `maxHands` hands. Frame can be any size.
  std::vector<Hand> update(const cv::Mat &bgr, int maxHands = 2);

private:
  struct Anchor {
    float cx, cy;
  };
  struct RawDet {
    float cx, cy, w, h, score;
    float kp[14]; // 7 keypoints × 2 (wrist, index_mcp … thumb_tip)
  };

  cv::dnn::Net palmNet_, lmNet_;
  std::vector<Anchor> anchors_; // 896 pre-generated anchors
  float detThr_, nmsThr_;
  bool loaded_ = false;

  void genAnchors();
  std::vector<RawDet> decodePalms(const std::vector<cv::Mat> &outs, int imgW,
                                  int imgH);
  std::vector<RawDet> nms(std::vector<RawDet> &dets);
  Hand runLandmark(const cv::Mat &bgr, const RawDet &d);

  // Returns [1, H, W, 3] float32 NHWC blob (MediaPipe model format)
  static cv::Mat nhwcBlob(const cv::Mat &bgr, int W, int H);
};
