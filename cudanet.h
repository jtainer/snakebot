// 
// Functions to create, train, and destroy a feedforward neural network on GPU
// 
// 2021, Jonathan Tainer
//	

#ifndef SYS_TO_DEV
#define SYS_TO_DEV 0
#endif

#ifndef DEV_TO_SYS
#define DEV_TO_SYS 1
#endif

#ifndef SYS_TO_SYS
#define SYS_TO_SYS 2
#endif

#ifndef SYSTEM_MEM
#define SYSTEM_MEM 0
#endif

#ifndef DEVICE_MEM
#define DEVICE_MEM 1
#endif

#ifndef CUDANET_H
#define CUDANET_H

#include "network.h"

/* Functions for creating, copying, and destroying layers in system and device memory */

void initLayer(Layer* sysLayer, int numOfInputs, int numOfNodes);

void deleteLayer(Layer* sysLayer);

void copyLayer(Layer* devLayer, Layer* sysLayer, int direction);

void cudaDeleteLayer(Layer* devLayer);

/* Forward propagation functions */

void forwardPass(Layer* layer, int numOfLayers, float* inputVector, float* outputVector);

/* Backprop functions */
void backwardPass(Layer* devLayer, int numOfLayers, float* inputVector, float* targetVector);

/* Network handling functions */

extern "C" Network createNetwork(int numOfInputs, int numOfOutputs, int numOfLayers, int nodesPerLayer);

//Network cudaCreateNetwork(int numOfInputs, int numOfOutputs, int numOfLayers, int nodesPerLayer);

Network copyNetwork(Network sourceNet, int destination);

void copyNetworkData(Network* destination, Network* source, int direction);

extern "C" void deleteNetwork(Network* net);

void cudaDeleteNetwork(Network* net);

void randomizeNetwork(Network net);

void mutateNetwork(Network net, float probability, float variance);

void saveNetwork(Network net, const char* file);

Network readNetwork(const char* file);



#endif
