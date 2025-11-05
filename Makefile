NVCC := nvcc
CXXFLAGS := -O2

.PHONY: all clean run help

all: hough

hough: houghBase.cu pgm.o
	$(NVCC) $(CXXFLAGS) houghBase.cu pgm.o -o hough

pgm.o: common/pgm.cpp common/pgm.h
	$(NVCC) -c common/pgm.cpp -o pgm.o

run: hough
	./hough

clean:
	-@rm -f hough hough.exe pgm.o

help:
	@echo Targets:
	@echo   all    - build the executable 'hough'
	@echo   run    - run the program (expects 'runway.pgm' in current dir)
	@echo   clean  - remove build artifacts
