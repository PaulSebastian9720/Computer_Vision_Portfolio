/*
 * ESP32-CAM — Servidor MJPEG para visión estéreo
 *
 * ┌─────────────────────────────────────────────────────┐
 * │  CAMBIAR ANTES DE FLASHEAR                          │
 * │  SSID     → nombre de la red WiFi                   │
 * │  PASSWORD → contraseña de la red WiFi               │
 * └─────────────────────────────────────────────────────┘
 *
 * La IP la asigna el router — se imprime por Serial al arrancar.
 * Stream:   http://<IP>/stream
 * Snapshot: http://<IP>/capture
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_wifi.h"
#include <WiFi.h>

const char *SSID = "Telecable-Fibra Naspud Vivar";
const char *PASSWORD = "#.TEFY9728.";

// Pinout AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// ─── MJPEG stream handler ─────────────────────────────
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = nullptr;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK)
    return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
      break;
    }

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res != ESP_OK) {
      esp_camera_fb_return(fb);
      break;
    }

    size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, fb->len);
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res != ESP_OK) {
      esp_camera_fb_return(fb);
      break;
    }

    res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK)
      break;
  }
  return res;
}

// ─── Snapshot handler ────────────────────────────────
static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition",
                     "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// ─── Servidor HTTP ────────────────────────────────────
static httpd_handle_t camera_httpd = nullptr;

void startServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 8;
  config.lru_purge_enable = true; // libera conexiones inactivas rápido

  httpd_uri_t stream_uri = {.uri = "/stream",
                            .method = HTTP_GET,
                            .handler = stream_handler,
                            .user_ctx = nullptr};
  httpd_uri_t capture_uri = {.uri = "/capture",
                             .method = HTTP_GET,
                             .handler = capture_handler,
                             .user_ctx = nullptr};

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
  }
}

// ─── setup() ─────────────────────────────────────────
void setup() {
  // CPU a máxima velocidad (240 MHz) — primera línea, antes de todo
  setCpuFrequencyMhz(240);

  Serial.begin(115200);
  Serial.println("\n[BOOT] ESP32-CAM stream");
  Serial.printf("[CPU] %d MHz\n", getCpuFrequencyMhz());

  // Inicializar cámara
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA; // 320x240
  config.jpeg_quality =
      20; // 20 en vez de 12: archivos ~30% más pequeños, más fps
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] Error 0x%x — reiniciando\n", err);
    delay(1000);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0);
  s->set_exposure_ctrl(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_dcw(s, 1);
  Serial.println("[CAM] OK");

  // WiFi: deshabilitar sleep ANTES de conectar (principal causa de lentitud)
  WiFi.setSleep(false);
  WiFi.begin(SSID, PASSWORD);
  Serial.printf("[WiFi] Conectando a '%s'", SSID);

  int intentos = 0;
  while (WiFi.status() != WL_CONNECTED && intentos < 30) {
    delay(500);
    Serial.print(".");
    intentos++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WiFi] Falló — reiniciando");
    delay(2000);
    ESP.restart();
  }

  // Deshabilitar power save del stack WiFi (reduce latencia ~50ms → <5ms)
  esp_wifi_set_ps(WIFI_PS_NONE);

  startServer();

  Serial.println();
  Serial.println("=====================================");
  Serial.printf("  IP:     %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("  RSSI:   %d dBm\n", WiFi.RSSI());
  Serial.printf("  Stream: http://%s/stream\n",
                WiFi.localIP().toString().c_str());
  Serial.println("=====================================");
}

// ─── loop() ──────────────────────────────────────────
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Perdida — reconectando...");
    WiFi.reconnect();
    int t = 0;
    while (WiFi.status() != WL_CONNECTED && t < 20) {
      delay(500);
      Serial.print(".");
      t++;
    }
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("\n[WiFi] Sin red — reiniciando");
      ESP.restart();
    }
    esp_wifi_set_ps(WIFI_PS_NONE);
    Serial.printf("\n[WiFi] Reconectado: %s\n",
                  WiFi.localIP().toString().c_str());
  }
  delay(5000);
}
