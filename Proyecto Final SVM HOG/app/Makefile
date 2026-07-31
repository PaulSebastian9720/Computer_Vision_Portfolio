.PHONY: all run run_video clean

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Wno-dev
	cmake --build build -j$$(nproc)

run: all
	./build/truck_trigger

# VIDEO=/ruta/al/video.mp4 para cambiar el video (por defecto CamionesFrontal.mp4, ya
# viene puesto adentro de main_video.cpp junto con el modelo y el umbral por defecto)
# THRESHOLD=0.5 para forzar un umbral distinto al que trae el .yml
DEFAULT_VIDEO := ../artifcats/videos/CamionesFrontal.mp4
run_video: all
ifeq ($(VIDEO),)
	./build/truck_trigger_video "$(DEFAULT_VIDEO)" $(THRESHOLD)
else
	./build/truck_trigger_video "$(VIDEO)" $(THRESHOLD)
endif

# Nuevas variables para el detector dual de camiones grandes
VIDEO_DUAL ?= ../artifcats/videos/trafico.webm
TH_FRONTAL ?= -1001.0
TH_LATERAL ?= -1001.0
MODEL_FRONT ?= "../attachments/detector_frontal.yml"
MODEL_LAT ?= "../attachments/detector_lateral.yml"

run_dual: all
	./build/truck_detector_dual "$(VIDEO_DUAL)" $(TH_FRONTAL) $(TH_LATERAL) $(MODEL_FRONT) $(MODEL_LAT)

clean:
	rm -rf build
