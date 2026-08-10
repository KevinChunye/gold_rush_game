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
# (The official sample uses `-O2 -march=native -fPIC -shared`; -march=native
# is fine when compiling on the contest server, whose CPU matches the judge
# machines, but we skip it so the .so also runs when built elsewhere.)
SOFLAGS = -shared -fPIC -static-libstdc++ -static-libgcc

player.so: src/player.cpp src/game_api.h src/constants.h
	$(CXX) $(CXXFLAGS) $(SOFLAGS) -o $@ src/player.cpp

test_runner: test/local_test.cpp src/game_api.h src/constants.h
	$(CXX) $(CXXFLAGS) -o $@ test/local_test.cpp -ldl

.PHONY: test clean
test: player.so test_runner
	./test_runner

clean:
	rm -f player.so test_runner
