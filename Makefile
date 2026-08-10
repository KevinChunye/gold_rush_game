# GoldRush 2.0 — build the submission (player.so) and the local smoke test.
#
#   make          -> player.so   (this is the file you upload)
#   make test     -> build + run the offline test harness
#   make clean
#
# NOTE: player.so must be a Linux x86_64 shared object. Build it on the
# contest dev server (or any Linux box) — a macOS build produces a Mach-O
# .dylib that the engine cannot load.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

# -static-libstdc++/-static-libgcc: the engine dlopen()s our library into
# its own process, so we avoid depending on whatever libstdc++ it ships.
SOFLAGS = -shared -fPIC -static-libstdc++ -static-libgcc

player.so: src/player.cpp src/game_api.h
	$(CXX) $(CXXFLAGS) $(SOFLAGS) -o $@ src/player.cpp

test_runner: test/local_test.cpp src/game_api.h
	$(CXX) $(CXXFLAGS) -o $@ test/local_test.cpp -ldl

.PHONY: test clean
test: player.so test_runner
	./test_runner

clean:
	rm -f player.so test_runner
