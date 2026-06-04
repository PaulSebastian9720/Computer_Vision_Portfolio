#include "esp_camera.h"
#include <WiFi.h>

// ===================
// Select camera model
// ===================
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
#include "camera_pins.h"

// ===========================
// Enter your WiFi credentials
// ===========================
const char *ssid     = "Telecable-Fibra Naspud Vivar";
const char *password = "#.TEFY9728.";

// ─── STEREO MANUAL EXPOSURE PARAMS ───────────────────────────────────────────
// Flash the SAME values to BOTH cameras.
// Procedure: let one camera run in auto mode, read the values it settled on
// under the lab lighting, then hardcode those values here for both units.
static constexpr int FIXED_AGC_GAIN  = 5;    // 0–30
static constexpr int FIXED_AEC_VALUE = 400;  // 0–1200
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

  // 10 MHz: mejor estabilidad de paquetes en red vs 20 MHz.
  // Bajar a 8 MHz si siguen apareciendo frames corruptos.
  config.xclk_freq_hz = 10000000;

  config.frame_size   = FRAMESIZE_VGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;  // 10 vs 14: menos compresión = menos artefactos de bloque
  config.fb_count     = 2;

  // Si no hay PSRAM disponible, usar DRAM con buffer único
  if (config.pixel_format == PIXFORMAT_JPEG && !psramFound()) {
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count    = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();

  // Correcciones específicas para el sensor OV3660
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

  // ─── PARIDAD LUMÍNICA ESTÉREO ─────────────────────────────────────────────
  // Deshabilitar TODOS los controles automáticos para que ambas cámaras
  // produzcan la misma exposición y no introduzcan discrepancias fotométricas
  // que engañen al algoritmo de block matching.
  s->set_exposure_ctrl(s, 0);   // Deshabilitar AEC
  s->set_gain_ctrl(s, 0);       // Deshabilitar AGC
  s->set_whitebal(s, 0);        // Deshabilitar balance de blancos automático
  s->set_awb_gain(s, 0);        // Deshabilitar ganancia AWB

  // Fijar valores manuales idénticos en ambas unidades
  s->set_aec_value(s, FIXED_AEC_VALUE);
  s->set_agc_gain(s, FIXED_AGC_GAIN);
  s->set_wb_mode(s, 0);         // Modo de iluminación fijo (0 = Auto desactivado)
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
  Serial.println("\nWiFi connected");

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

void loop() {
  delay(10000);
}
