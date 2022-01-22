nvcc -c cudanet.cu matmul.cu
g++ -c train.cpp snake.cpp
g++ -L/usr/local/cuda/lib64 train.o snake.o cudanet.o matmul.o -lcudart
