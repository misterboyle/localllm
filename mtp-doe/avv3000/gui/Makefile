# gui Makefile — r_gui rung wrapper over CMake (Qt6 offscreen).
# score.sh only invokes this when qmake exists (env.sh tier `gui`).
QT ?= $(shell brew --prefix)/opt/qt

.PHONY: all configure run clean

all: run

configure:
	cmake -S . -B build -DCMAKE_PREFIX_PATH=$(QT) -DCMAKE_BUILD_TYPE=Release

run: configure
	cmake --build build
	QT_QPA_PLATFORM=offscreen ctest --test-dir build --output-on-failure

clean:
	rm -rf build
