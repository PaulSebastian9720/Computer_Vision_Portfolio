.PHONY: all run_video run_video_dual_bot run_camera_dual_bot run_dual clean

VIDEO ?= ../targets/camiones_nuevovision/videos/CamionesFrontal.mp4
THRESHOLD ?= auto
DETECTOR ?= ../targets/camiones_nuevovision/models/detector_lateral_latest.yml
MIN_CENTER_Y ?= 0.30
MAX_CENTER_Y ?= 1.0
MAX_AREA ?= 1.0
MIN_AREA ?= 0.0
PREPROCESS ?= none
ROI_MIN_X ?= 0.0
ROI_MAX_X ?= 1.0
LABEL ?= camion
COLOR_FILTER ?= none
HISTORY_LEN ?= 6
CONFIRM_HITS ?= 0
IOU_MIN ?= 0.3
SERVER_URL ?= http://localhost:8000/detect
ALERT_COOLDOWN ?= 1.0
CAMERA_INDEX ?= 0

VIDEO_DUAL ?= ../targets/camiones_nuevovision/videos/CamionesColombia.mkv
TH_FRONTAL ?= 0.60
TH_LATERAL ?= 0.80
MODEL_FRONT ?= ../targets/camiones_nuevovision/models/detector_frontal.yml
MODEL_LAT ?= ../targets/camiones_nuevovision/models/detector_lateral_latest.yml

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Wno-dev
	cmake --build build --parallel

run_video: all
	./build/truck_trigger_video "$(VIDEO)" "$(THRESHOLD)" "$(DETECTOR)" "$(MIN_CENTER_Y)" "$(MAX_AREA)" "$(PREPROCESS)" "$(ROI_MIN_X)" "$(ROI_MAX_X)" "$(LABEL)" "$(MIN_AREA)" "$(COLOR_FILTER)" "$(MAX_CENTER_Y)" "$(HISTORY_LEN)" "$(CONFIRM_HITS)" "$(IOU_MIN)" "$(SERVER_URL)" "$(ALERT_COOLDOWN)"

run_video_dual_bot: all
	./build/truck_trigger_video "$(VIDEO_DUAL)" "$(TH_LATERAL)" "$(MODEL_LAT)" "$(MIN_CENTER_Y)" "$(MAX_AREA)" "$(PREPROCESS)" "$(ROI_MIN_X)" "$(ROI_MAX_X)" "$(LABEL)" "$(MIN_AREA)" "$(COLOR_FILTER)" "$(MAX_CENTER_Y)" "$(HISTORY_LEN)" "$(CONFIRM_HITS)" "$(IOU_MIN)" "$(SERVER_URL)" "$(ALERT_COOLDOWN)" "$(MODEL_FRONT)" "$(TH_FRONTAL)"

run_camera_dual_bot: all
	./build/truck_trigger_video "camera:$(CAMERA_INDEX)" "$(TH_LATERAL)" "$(MODEL_LAT)" "$(MIN_CENTER_Y)" "$(MAX_AREA)" "$(PREPROCESS)" "$(ROI_MIN_X)" "$(ROI_MAX_X)" "$(LABEL)" "$(MIN_AREA)" "$(COLOR_FILTER)" "$(MAX_CENTER_Y)" "$(HISTORY_LEN)" "$(CONFIRM_HITS)" "$(IOU_MIN)" "$(SERVER_URL)" "$(ALERT_COOLDOWN)" "$(MODEL_FRONT)" "$(TH_FRONTAL)"

run_dual: all
	./build/truck_detector_dual "$(VIDEO_DUAL)" "$(TH_FRONTAL)" "$(TH_LATERAL)" "$(MODEL_FRONT)" "$(MODEL_LAT)"

clean:
	rm -rf build
