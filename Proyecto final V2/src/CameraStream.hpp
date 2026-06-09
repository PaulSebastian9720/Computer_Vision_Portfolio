#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

class CameraStream {
public:
    CameraStream(const std::string& url, int id);
    ~CameraStream();

    void start();
    void stop();

    // Non-blocking: returns true only if a new frame arrived since last call.
    bool getFrame(cv::Mat& frame, std::chrono::steady_clock::time_point& ts);

private:
    void captureLoop();

    std::string url_;
    int         id_;

    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable cv_;

    cv::Mat                               latestFrame_;
    std::chrono::steady_clock::time_point latestTs_;
    bool                                  hasNew_{false};

    std::atomic<bool> running_{false};
};
