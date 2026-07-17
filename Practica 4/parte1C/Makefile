.PHONY: all run-cpu run-gpu clean

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Wno-dev
	cmake --build build -j$$(nproc)

run-cpu: all
	./build/preprocesamiento_cpu

run-gpu: all
	./build/preprocesamiento_gpu

clean:
	rm -rf build
