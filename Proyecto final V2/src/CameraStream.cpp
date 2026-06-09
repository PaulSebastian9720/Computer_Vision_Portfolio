#include "CameraStream.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

static bool parseUrl(const std::string& url,
                     std::string& host, int& port, std::string& path) {
    auto p = url.find("://");
    if (p == std::string::npos) return false;
    std::string rest = url.substr(p + 3);
    auto sl = rest.find('/');
    path    = (sl != std::string::npos) ? rest.substr(sl) : "/";
    auto hp = (sl != std::string::npos) ? rest.substr(0, sl) : rest;
    auto colon = hp.find(':');
    if (colon != std::string::npos) {
        host = hp.substr(0, colon);
        port = std::stoi(hp.substr(colon + 1));
    } else { host = hp; port = 80; }
    return true;
}

CameraStream::CameraStream(const std::string& url, int id) : url_(url), id_(id) {}
CameraStream::~CameraStream() { stop(); }

void CameraStream::start() {
    running_ = true;
    thread_  = std::thread(&CameraStream::captureLoop, this);
}

void CameraStream::stop() {
    running_ = false;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool CameraStream::getFrame(cv::Mat& frame,
                              std::chrono::steady_clock::time_point& ts) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!hasNew_) return false;
    frame  = latestFrame_.clone();
    ts     = latestTs_;
    hasNew_ = false;
    return true;
}

void CameraStream::captureLoop() {
    std::string host, path;
    int port = 80;
    if (!parseUrl(url_, host, port, path)) {
        std::cerr << "[Cam" << id_ << "] Bad URL: " << url_ << "\n";
        return;
    }

    while (running_) {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) { std::this_thread::sleep_for(std::chrono::seconds(1)); continue; }

        struct timeval tv{5, 0};
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&tv), sizeof(tv));

        struct hostent* he = gethostbyname(host.c_str());
        if (!he) {
            close(sockfd);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(static_cast<uint16_t>(port));
        memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

        if (connect(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            close(sockfd);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        std::string req = "GET " + path + " HTTP/1.1\r\nHost: " + host +
                          "\r\nConnection: close\r\n\r\n";
        send(sockfd, req.c_str(), req.size(), 0);
        std::cout << "[Cam" << id_ << "] Connected to " << host << "\n";

        std::vector<uint8_t> accum;
        accum.reserve(64 * 1024);
        uint8_t prev = 0;
        bool    inJpeg = false;
        char    buf[16384];

        while (running_) {
            int n = recv(sockfd, buf, sizeof(buf), 0);
            if (n <= 0) { std::cerr << "[Cam" << id_ << "] Reconnecting...\n"; break; }

            for (int i = 0; i < n; ++i) {
                const uint8_t b = static_cast<uint8_t>(buf[i]);
                if (!inJpeg) {
                    if (prev == 0xFF && b == 0xD8) {
                        inJpeg = true;
                        accum.clear();
                        accum.push_back(0xFF);
                        accum.push_back(0xD8);
                    }
                } else {
                    accum.push_back(b);
                    if (prev == 0xFF && b == 0xD9) {
                        cv::Mat frame = cv::imdecode(accum, cv::IMREAD_COLOR);
                        if (!frame.empty()) {
                            std::lock_guard<std::mutex> lk(mutex_);
                            latestFrame_ = frame;
                            latestTs_    = std::chrono::steady_clock::now();
                            hasNew_      = true;
                            cv_.notify_one();
                        }
                        inJpeg = false;
                        accum.clear();
                    }
                }
                prev = b;
            }
        }
        close(sockfd);
        if (running_) std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}
