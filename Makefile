
FLAGS= -L/usr/local/cuda/lib64 -I/usr/local/cuda-9.2/targets/x86_64-linux/include -lcuda -lcudart
ANIMLIBS= -lglut -lGL

all: ex2

ex2: interface.o gpu_main.o animate.o
	g++ -o ex2 interface.o gpu_main.o animate.o $(FLAGS) $(ANIMLIBS)

interface.o: interface.cpp interface.h params.h animate.h animate.cu
	g++ -w -c interface.cpp $(FLAGS)

gpu_main.o: gpu_main.cu gpu_main.h params.h
	nvcc -w -c gpu_main.cu

animate.o: animate.cu animate.h gpu_main.h params.h
	nvcc -w -c animate.cu

clean:
	rm interface.o;
	rm gpu_main.o;
	rm animate.o;
