.PHONY: all run ar clean

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Wno-dev
	cmake --build build -j$$(nproc)

run: all
	./build/stereo_v2

ar: all
	./build/ar_filter

clean:
	rm -rf build
