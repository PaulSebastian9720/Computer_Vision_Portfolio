#include "CameraStream.hpp"
#include <iostream>
#include <chrono>

CameraStream::CameraStream(const std::string& url, int id)
    : url_(url), id_(id) {}

CameraStream::~CameraStream() {
    stop();
}

void CameraStream::start() {
    running_ = true;
    thread_  = std::thread(&CameraStream::captureLoop, this);
}

void CameraStream::stop() {
    running_ = false;
    cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
}

bool CameraStream::getFrame(cv::Mat& frame,
                             std::chrono::steady_clock::time_point& timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hasNewFrame_) return false;
    frame       = latestFrame_.clone();
    timestamp   = latestTimestamp_;
    hasNewFrame_ = false;
    return true;
}

void CameraStream::captureLoop() {
    cv::VideoCapture cap;

    while (running_) {
        if (!cap.isOpened()) {
            std::cout << "[Cam " << id_ << "] Connecting to " << url_ << "\n";
            cap.open(url_, cv::CAP_FFMPEG);
            if (!cap.isOpened()) {
                std::cerr << "[Cam " << id_ << "] Connection failed, retrying in 1s...\n";
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            // Minimize internal buffer so we always get the latest frame
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            std::cout << "[Cam " << id_ << "] Connected.\n";
        }

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[Cam " << id_ << "] Stream lost, reconnecting...\n";
            cap.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            latestFrame_     = frame;
            latestTimestamp_ = std::chrono::steady_clock::now();
            hasNewFrame_     = true;
        }
        cv_.notify_one();
    }
}
