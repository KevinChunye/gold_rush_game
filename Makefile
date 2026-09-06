# GoldRush 2.0 bot — build and test.
#
#   make        -> player.so   (the tournament artifact the engine dlopens)
#   make test   -> offline harness: loads player.so the same way the real
#                  engine does and runs scripted scenarios + a latency bench
#   make clean

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

player.so: src/player.cpp src/game_api.h src/constants.h
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ src/player.cpp

test_runner: test/local_test.cpp src/game_api.h src/constants.h
	$(CXX) $(CXXFLAGS) -o $@ test/local_test.cpp -ldl

.PHONY: test clean
test: player.so test_runner
	./test_runner

clean:
	rm -f player.so test_runner
