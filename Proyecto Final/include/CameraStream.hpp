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

    // Non-blocking: fills frame+timestamp and returns true only if a new frame
    // arrived since the last call. Returns false if no new frame is ready yet.
    bool getFrame(cv::Mat& frame, std::chrono::steady_clock::time_point& timestamp);

private:
    void captureLoop();

    std::string url_;
    int         id_;

    std::thread              thread_;
    std::mutex               mutex_;
    std::condition_variable  cv_;

    cv::Mat                               latestFrame_;
    std::chrono::steady_clock::time_point latestTimestamp_;
    bool                                  hasNewFrame_{false};

    std::atomic<bool> running_{false};
};
