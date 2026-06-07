#include "esp_camera.h"
#include <WiFi.h>

// ============================================================
// Board: AI-Thinker ESP32-CAM (pines originales via camera_pins.h)
// Módulo de cámara: OV2640 (ya sea el original o el del XIAO Sense)
// ============================================================
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// ===========================
// Credenciales WiFi
// ===========================
const char *ssid     = "Telecable-Fibra Naspud Vivar";
const char *password = "#.TEFY9728.";

// ─── PARÁMETROS DE EXPOSICIÓN MANUAL ESTÉREO ─────────────────────────────────
// Flashear los MISMOS valores en las DOS unidades.
static constexpr int FIXED_AGC_GAIN  = 5;    // 0–30
static constexpr int FIXED_AEC_VALUE = 600;  // 0–1200  (≈3 ciclos de 60Hz en VGA 15fps → menos banding)
// ─────────────────────────────────────────────────────────────────────────────

void startCameraServer();
void setupLedFlash(int pin);

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  // 8 MHz: reduce ruido eléctrico del sensor → menos rayas horizontales.
  config.xclk_freq_hz = 8000000;

  config.frame_size   = FRAMESIZE_VGA;    // 640×480
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 8;
  config.fb_count     = 2;

  if (config.pixel_format == PIXFORMAT_JPEG && !psramFound()) {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count    = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();

  // OV3660 specific fixes (si aplica)
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }

  // ─── PARIDAD LUMÍNICA ESTÉREO ─────────────────────────────────────────────
  // Desactivar TODOS los automáticos — ambas cámaras deben ver idéntico.
  s->set_exposure_ctrl(s, 0);
  s->set_gain_ctrl(s, 0);
  s->set_whitebal(s, 0);
  s->set_awb_gain(s, 0);

  s->set_aec_value(s, FIXED_AEC_VALUE);
  s->set_agc_gain(s, FIXED_AGC_GAIN);
  s->set_wb_mode(s, 0);

  s->set_brightness(s, 0);
  s->set_contrast(s, 0);
  s->set_saturation(s, 0);
  s->set_sharpness(s, 0);

  // Aplicar AEC_VALUE dos veces: el driver OV2640 a veces ignora el primer write
  // porque los registros se actualizan en el siguiente VSYNC.
  delay(100);
  s->set_aec_value(s, FIXED_AEC_VALUE);
  s->set_agc_gain(s, FIXED_AGC_GAIN);
  // ─────────────────────────────────────────────────────────────────────────

#if defined(LED_GPIO_NUM)
  setupLedFlash(LED_GPIO_NUM);
#endif

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi conectado");

  startCameraServer();

  Serial.print("Camara lista: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
}

void loop() {
  delay(10000);
}
