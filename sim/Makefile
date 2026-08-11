# sim/Makefile — headless GoldRush 2.0 match simulator + smoke-test bot.
#
#   make          -> simulator + dummy.so
#   make smoke    -> run the standard smoke match (two dummies, mirrored)
#   make clean

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

all: simulator dummy.so

simulator: simulator.cpp ../src/game_api.h ../src/constants.h
	$(CXX) $(CXXFLAGS) -o $@ simulator.cpp -ldl

dummy.so: dummy_player.cpp ../src/game_api.h
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ dummy_player.cpp

smoke: simulator dummy.so
	./simulator dummy.so dummy.so --games 2 --seed 7 --mirror

clean:
	rm -f simulator dummy.so

.PHONY: all smoke clean
