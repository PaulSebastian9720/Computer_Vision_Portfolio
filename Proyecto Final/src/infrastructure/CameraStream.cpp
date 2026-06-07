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

// ── URL parser ────────────────────────────────────────────────────────────────
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
    } else {
        host = hp;
        port = 80;
    }
    return true;
}

// ── CameraStream ──────────────────────────────────────────────────────────────

CameraStream::CameraStream(const std::string& url, int id)
    : url_(url), id_(id) {}

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
                             std::chrono::steady_clock::time_point& timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hasNewFrame_) return false;
    frame        = latestFrame_.clone();
    timestamp    = latestTimestamp_;
    hasNewFrame_ = false;
    return true;
}

// ── Custom MJPEG reader — JPEG marker scan ───────────────────────────────────
//
// Instead of parsing HTTP chunked-encoding or MJPEG part headers (which vary
// between ESP32-CAM firmware versions), we scan the raw TCP byte stream for
// JPEG SOI (FF D8 FF) and EOI (FF D9) markers.  This works regardless of
// whether the server uses HTTP/1.0, HTTP/1.1 chunked, or any boundary format.
//
// JPEG byte-stuffing guarantees that FF D9 cannot appear inside compressed
// data (any FF in the entropy-coded stream is followed by 00, not D9), so the
// EOI scan is unambiguous.

void CameraStream::captureLoop() {
    std::string host, path;
    int port = 80;
    if (!parseUrl(url_, host, port, path)) {
        std::cerr << "[Cam " << id_ << "] Bad URL: " << url_ << "\n";
        return;
    }

    while (running_) {
        // ── 1. Create socket ──────────────────────────────────────────────────
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        struct timeval tv{5, 0};   // 5-second receive timeout
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&tv), sizeof(tv));

        // ── 2. Resolve and connect ────────────────────────────────────────────
        struct hostent* he = gethostbyname(host.c_str());
        if (!he) {
            std::cerr << "[Cam " << id_ << "] DNS failed for " << host << "\n";
            close(sockfd);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(static_cast<uint16_t>(port));
        memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);

        std::cout << "[Cam " << id_ << "] Connecting to "
                  << host << ":" << port << path << "\n";

        if (connect(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[Cam " << id_ << "] connect() failed\n";
            close(sockfd);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        // ── 3. Send HTTP GET ──────────────────────────────────────────────────
        std::string req = "GET " + path + " HTTP/1.1\r\n"
                          "Host: " + host + "\r\n"
                          "Connection: close\r\n\r\n";
        if (send(sockfd, req.c_str(), req.size(), 0) < 0) {
            close(sockfd);
            continue;
        }

        std::cout << "[Cam " << id_ << "] Connected, reading frames...\n";

        // ── 4. Scan raw TCP stream for JPEG markers ───────────────────────────
        // We accumulate bytes starting at each FF D8 FF (SOI) and stop at
        // FF D9 (EOI).  HTTP headers, MJPEG boundaries, and chunked-encoding
        // size lines are all transparently skipped because none of them contain
        // the SOI byte sequence.
        std::vector<uint8_t> accum;
        accum.reserve(64 * 1024);
        uint8_t prev = 0;
        bool    inJpeg = false;
        char    buf[16384];

        while (running_) {
            int n = recv(sockfd, buf, sizeof(buf), 0);
            if (n <= 0) {
                std::cerr << "[Cam " << id_ << "] Stream ended, reconnecting...\n";
                break;
            }

            for (int i = 0; i < n; i++) {
                const uint8_t b = static_cast<uint8_t>(buf[i]);

                if (!inJpeg) {
                    // Wait for FF D8 FF (JPEG SOI)
                    if (prev == 0xFF && b == 0xD8) {
                        inJpeg = true;
                        accum.clear();
                        accum.push_back(0xFF);
                        accum.push_back(0xD8);
                    }
                } else {
                    accum.push_back(b);

                    // FF D9 = JPEG EOI — complete frame
                    if (prev == 0xFF && b == 0xD9) {
                        cv::Mat frame = cv::imdecode(accum, cv::IMREAD_COLOR);
                        if (!frame.empty()) {
                            std::lock_guard<std::mutex> lock(mutex_);
                            latestFrame_     = frame;
                            latestTimestamp_ = std::chrono::steady_clock::now();
                            hasNewFrame_     = true;
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
        if (running_)
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}
