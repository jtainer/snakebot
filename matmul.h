// 
// A kernel that multiplies two vectors and applies the sigmoid activation function
//
// 2021, Jonathan Tainer
//


#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01
#endif


#ifndef MATMUL_H
#define MATMUL_H

#include <cuda.h>
#include "network.h"


//
// Forward pass functions
//

__global__
void forwardKernel(float* input, float* weights, float* output, const int numOfInputs, const int numOfNodes);


//
// Back propagation functions
//

__global__
void outputLayerDelta(Layer outputLayer, float* target);

__global__
void hiddenLayerDelta(Layer hiddenLayer, Layer nextLayer);

__global__
void updateWeights(Layer devLayer, float* inputVector);

#endif
