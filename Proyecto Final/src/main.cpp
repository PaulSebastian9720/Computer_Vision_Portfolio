#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "CameraStream.hpp"
#include "DesktopUI.hpp"
#include "Stabilizer.hpp"
#include "StereoService.hpp"

// LEFT debe ser la cámara físicamente a la IZQUIERDA mirando desde detrás.
// Si disparidad sale toda negra/inválida → swap estas dos líneas.
static constexpr char CAM_LEFT_URL[]  = "http://192.168.0.117:81/stream";
static constexpr char CAM_RIGHT_URL[] = "http://192.168.0.118:81/stream";
static constexpr char CALIB_YAML[] = "src/parametros_estereo.yml";

static constexpr int SYNC_THRESHOLD_MS = 250;  // VGA WiFi streams ~10fps; pairs can differ up to ~100ms naturally

int main() {
  CameraStream camL(CAM_LEFT_URL, 0);
  CameraStream camR(CAM_RIGHT_URL, 1);
  StereoService stereo(CALIB_YAML);

  camL.start();
  camR.start();

  cv::Mat frameL, frameR;

  // Initialise to now so the first sync check passes even before both
  // cameras have delivered a frame (avoids year-1970 timestamp diff).
  auto tsL = std::chrono::steady_clock::now();
  auto tsR = std::chrono::steady_clock::now();

  Stabilizer kalman;
  std::deque<float> zBuf;     // temporal median buffer
  static constexpr int Z_BUF = 9;
  cv::Mat dispColor;
  float zRawCm    = 0.0f;
  float zKalmanCm = 0.0f;
  int debugTick   = 0;

  std::cout << "Stereo vision running. Press ESC or 'q' to quit.\n";

  while (true) {
    bool newL = camL.getFrame(frameL, tsL);
    bool newR = camR.getFrame(frameR, tsR);

    if ((newL || newR) && !frameL.empty() && !frameR.empty() &&
        stereo.isCalibrated()) {

      auto diffMs = std::abs(
          std::chrono::duration_cast<std::chrono::milliseconds>(tsL - tsR)
              .count());

      if (diffMs <= SYNC_THRESHOLD_MS) {
        // Downscale to calibration resolution (320×240) before stereo processing.
        // Firmware streams VGA; calibration was done at QVGA — avoid map mismatch.
        cv::Mat procL, procR;
        cv::resize(frameL, procL, cv::Size(320, 240));
        cv::resize(frameR, procR, cv::Size(320, 240));

        cv::Mat leftRect = stereo.rectifyLeft(procL);
        cv::Mat rightRect = stereo.rectifyRight(procR);

        cv::Mat leftGray, rightGray;
        cv::cvtColor(leftRect, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightRect, rightGray, cv::COLOR_BGR2GRAY);

        // WLS-filtered disparity (numDisp=128 → covers 23 cm to ∞)
        cv::Mat dispRaw = stereo.process(leftGray, rightGray);

        // Escala fija: numDisp*16 = valor máximo posible en CV_16S ×16.
        // NORM_MINMAX se ve afectado por outliers; escala fija es consistente
        // → el mismo objeto a la misma distancia siempre tiene el mismo tono.
        cv::Mat dispGray;
        cv::threshold(dispRaw, dispGray, 0, 0, cv::THRESH_TOZERO);
        dispGray.convertTo(dispGray, CV_8U, 255.0 / (stereo.numDisp() * 16));
        cv::cvtColor(dispGray, dispColor, cv::COLOR_GRAY2BGR);

        cv::Mat disp32F;
        dispRaw.convertTo(disp32F, CV_32F, 1.0 / 16.0);
        cv::Mat pts3D;
        cv::reprojectImageTo3D(disp32F, pts3D, stereo.getQ(), true);

        // Sample 41×41 region around centre
        const int cy = leftGray.rows / 2;
        const int cx = leftGray.cols / 2;
        const int half = 20;
        std::vector<float> samples;
        float rawCenterZ = pts3D.at<cv::Point3f>(cy, cx).z;

        for (int dy = -half; dy <= half; ++dy)
          for (int dx = -half; dx <= half; ++dx) {
            int ry = cy + dy, rx = cx + dx;
            if (ry < 0 || ry >= pts3D.rows || rx < 0 || rx >= pts3D.cols)
              continue;
            float z = pts3D.at<cv::Point3f>(ry, rx).z;
            if (z < 0)
              z = -z;
            if (std::isfinite(z) && z > 10.0f && z < 5000.0f)
              samples.push_back(z);
          }

        if (++debugTick % 30 == 0) {
          float rawDisp = disp32F.at<float>(cy, cx);
          std::cout << "[DEBUG] disp=" << rawDisp
                    << "px  rawZ=" << rawCenterZ
                    << "mm  valid=" << samples.size()
                    << "/1681 (" << 100 * samples.size() / 1681 << "%)\n";
        }

        // Need at least 50 valid pixels (3% of 41×41) to trust the measurement.
        if (samples.size() >= 50) {
          std::sort(samples.begin(), samples.end());
          float medMm = samples[samples.size() / 2];
          zRawCm = medMm / 10.0f;

          // Temporal median: buffer last Z_BUF frame medians, feed their median to Kalman.
          // This rejects single-frame outliers before the Kalman even sees them.
          zBuf.push_back(medMm);
          if (static_cast<int>(zBuf.size()) > Z_BUF) zBuf.pop_front();

          std::vector<float> tmp(zBuf.begin(), zBuf.end());
          std::sort(tmp.begin(), tmp.end());
          float temporalMed = tmp[tmp.size() / 2];

          // Reset Kalman if the scene changed drastically (>40% jump from current estimate).
          if (kalman.isInitialized() &&
              std::abs(temporalMed / 10.0f - zKalmanCm) > zKalmanCm * 0.30f) {
            kalman.reset();
            zBuf.clear();
          }
          zKalmanCm = kalman.update(temporalMed) / 10.0f;
        }
      }
    }

    if (!frameL.empty()) {
      // Show raw VGA frame in camera panel (1:1 with 640×480 panel, no upscaling).
      // Stereo processing runs on the QVGA downscale; display uses the full sensor output.
      DesktopUI::render(frameL, dispColor, zRawCm, zKalmanCm);
    }

    int key = cv::waitKey(1);
    if (key == 27 || key == 'q')
      break;
  }

  camL.stop();
  camR.stop();
  cv::destroyAllWindows();
  return 0;
}
