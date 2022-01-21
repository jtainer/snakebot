// 
// Functions to create, train, and destroy a feed forward neural network on GPU
// 
// 2021, Jonathan Tainer
// 

#include "cudanet.h"
#include "matmul.h"
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void initLayer(Layer* sysLayer, int numOfInputs, int numOfNodes) {
	
	// Allocate memory for weight matrix
	sysLayer->weightMatrix = (float*)malloc(numOfInputs * numOfNodes * sizeof(float));

//	// Initialize weight matrix with random values between -1.f and 1.f
//	float multiplier = 2.f / RAND_MAX;
//	for (int i = 0; i < numOfInputs * numOfNodes; i++)
//		sysLayer->weightMatrix[i] = (rand() * multiplier) - 1.f;

	// Allocate memory for output vector
	sysLayer->outputVector = (float*)malloc((numOfNodes + 1) * sizeof(float));
	
	// Initialize the bias in the output vector to 1.f
	sysLayer->outputVector[numOfNodes] = 1.f; /* bias input */

	// Allocate memory for deltas
	sysLayer->delta = (float*)malloc(sizeof(float) * numOfNodes);

	// Note the weight and node counts for this layer
	sysLayer->numOfNodes = numOfNodes;
	sysLayer->weightsPerNode = numOfInputs;
}

/*
void cudaInitLayer(Layer* devLayer, int numOfInputs, int numOfNodes) {
	
	// Create a temporary layer in system memory
	Layer* tmpLayer;
	initLayer(tmpLayer, numOfInputs, numOfNodes);
	
	// Create a copy of temporary layer in VRAM
	copyLayer(devLayer, tmpLayer, SYS_TO_DEV);
	
	// Delete temporary layer from system memory
	deleteLayer(tmpLayer);
}
*/

void deleteLayer(Layer* sysLayer) {

	// Deallocate memory for weight matrix, output vector, and delta vector
	free(sysLayer->weightMatrix);
	free(sysLayer->outputVector);
	free(sysLayer->delta);
	
	// Point weight matrix and output vector to null
	sysLayer->weightMatrix = NULL;
	sysLayer->outputVector = NULL;
	sysLayer->delta = NULL;
	
	// Note that the layer contains no weights or nodes
	sysLayer->numOfNodes = 0;
	sysLayer->weightsPerNode = 0;
}

// This function assumes that the destination layer does not have memory allocated to it
// Calling copyLayer on a destination layer which has memory allocated to it will result in a memory leak
void copyLayer(Layer* devLayer, Layer* sysLayer, int direction) {

	switch (direction) {

	case SYS_TO_DEV:

		// Copy node and weight count from system layer to device layer
		devLayer->numOfNodes = sysLayer->numOfNodes;
		devLayer->weightsPerNode = sysLayer->weightsPerNode;

		// Allocate an appropriate amount of video memory		
		cudaMalloc((void**)&devLayer->weightMatrix, sizeof(float) * devLayer->numOfNodes * devLayer->weightsPerNode);
		cudaMalloc((void**)&devLayer->outputVector, sizeof(float) * (devLayer->numOfNodes + 1));
		cudaMalloc((void**)&devLayer->delta, sizeof(float) * devLayer->numOfNodes);
		
		// Copy weight matrix and output vector from system memory to video memory
		cudaMemcpy(devLayer->weightMatrix, sysLayer->weightMatrix, sizeof(float) * devLayer->numOfNodes * devLayer->weightsPerNode, cudaMemcpyHostToDevice);
		cudaMemcpy(devLayer->outputVector, sysLayer->outputVector, sizeof(float) * (devLayer->numOfNodes + 1), cudaMemcpyHostToDevice);
		break;

	case DEV_TO_SYS:

		//Copy node and weight count from device layer to system layer
		sysLayer->numOfNodes = devLayer->numOfNodes;
		sysLayer->weightsPerNode = devLayer->weightsPerNode;
		
		// Allocate an appropriate amount of system memory
		sysLayer->weightMatrix = (float*)malloc(sysLayer->numOfNodes * sysLayer->weightsPerNode * sizeof(float));
		sysLayer->outputVector = (float*)malloc((sysLayer->numOfNodes + 1) * sizeof(float));

		// Copy weight matrix and output vector from video memory to system memory
		cudaMemcpy(sysLayer->weightMatrix, devLayer->weightMatrix, sizeof(float) * sysLayer->numOfNodes * sysLayer->weightsPerNode, cudaMemcpyDeviceToHost);
		cudaMemcpy(sysLayer->outputVector, devLayer->outputVector, sizeof(float) * (sysLayer->numOfNodes + 1), cudaMemcpyDeviceToHost);
		break;
	}
}

void cudaDeleteLayer(Layer* devLayer) {
	
	// Deallocate video memory for weight matrix and output vector
	cudaFree(devLayer->weightMatrix);
	cudaFree(devLayer->outputVector);
	cudaFree(devLayer->delta);
	
	// Point weight matrix and output vector to null
	devLayer->weightMatrix = NULL;
	devLayer->outputVector = NULL;
	devLayer->delta = NULL;
	
	// Note that the layer contains no weights or nodes
	devLayer->numOfNodes = 0;
	devLayer->weightsPerNode = 0;
}

// devLayer must point to an array of layers in video memory, not system memory
// inputVector and outputVector must point to system memory
void forwardPass(Layer* devLayer, int numOfLayers, float* inputVector, float* outputVector) {
	
	// Copy input vector to video memory
	float* devInputVector;
	cudaMalloc((void**)&devInputVector, sizeof(float) * devLayer[0].weightsPerNode);
	cudaMemcpy(devInputVector, inputVector, sizeof(float) * devLayer[0].weightsPerNode, cudaMemcpyHostToDevice);

	// Run the kernel for the first layer, using the input vector provided by the function call
	forwardKernel<<<(devLayer[0].numOfNodes / 256) + 1, 256>>>
		(devInputVector, devLayer[0].weightMatrix, devLayer[0].outputVector, devLayer[0].weightsPerNode, devLayer[0].numOfNodes);

	// Deallocate video memory for input vector
	cudaFree(devInputVector);

	// Iterate through the layers of the network starting with the second layer
	// Run the kernel for each iteration, using the output of the previous layer as the input for the current layer
	for (int n = 1; n < numOfLayers; n++) {
		forwardKernel<<<(devLayer[n].numOfNodes / 256) + 1, 256>>>
			(devLayer[n - 1].outputVector, devLayer[n].weightMatrix, devLayer[n].outputVector, devLayer[n].weightsPerNode, devLayer[n].numOfNodes);
	}

	// Copy final output vector to system memory
	cudaMemcpy(outputVector, devLayer[numOfLayers - 1].outputVector, sizeof(float) * devLayer[numOfLayers - 1].numOfNodes, cudaMemcpyDeviceToHost);

}

// devLayer must point to an array of layers in video memory, not system memory
// inputVector and targetVector must point to system memory
void backwardPass(Layer* devLayer, int numOfLayers, float* inputVector, float* targetVector) {

	// Copy inputVector and targetVector to video memory
	float* devInputVector;
	float* devTargetVector;
	cudaMalloc((void**)&devInputVector, sizeof(float) * devLayer[0].weightsPerNode);
	cudaMalloc((void**)&devTargetVector, sizeof(float) * devLayer[numOfLayers - 1].numOfNodes);
	cudaMemcpy(devInputVector, inputVector, sizeof(float) * devLayer[0].weightsPerNode, cudaMemcpyHostToDevice);
	cudaMemcpy(devTargetVector, targetVector, sizeof(float) * devLayer[numOfLayers - 1].numOfNodes, cudaMemcpyHostToDevice);

	// Compute deltas for output layer
	outputLayerDelta<<<(devLayer[numOfLayers - 1].numOfNodes / 256) + 1, 256>>>
		(devLayer[numOfLayers - 1], devTargetVector);

	// Compute deltas for remaining layers
	for (int i = numOfLayers - 2; i >= 0; i--) {
		hiddenLayerDelta<<<(devLayer[i].numOfNodes / 256) + 1, 256>>>
			(devLayer[i], devLayer[i + 1]);
	}
	
	// Update weights for input layer
	updateWeights<<<(devLayer[0].numOfNodes / 256) + 1, 256>>>
		(devLayer[0], devInputVector);
		
		
	// Update weights for remaining layers
	for (int i = 1; i < numOfLayers - 1; i++) {
		updateWeights<<<(devLayer[i].numOfNodes / 256) + 1, 256>>>
			(devLayer[i], devLayer[i - 1].outputVector);
	}

	// Deallocate video memory for input and target vectors
	cudaFree(devInputVector);
	cudaFree(devTargetVector);
}

Network createNetwork(int numOfInputs, int numOfOutputs, int numOfLayers, int nodesPerLayer) {
	
	Network net;
	net.numOfLayers = numOfLayers;
	
	// Allocate enough memory for all layers of the neural network
	net.layer = (Layer*)malloc(numOfLayers * sizeof(Layer));
	
	// Construct input layer
	initLayer(&net.layer[0], numOfInputs, nodesPerLayer);
	
	// Construct hidden layers
	for (int i = 1; i < numOfLayers - 1; i++) {
		initLayer(&net.layer[i], nodesPerLayer + 1, nodesPerLayer);
	}
	
	// Construct output layer
	initLayer(&net.layer[numOfLayers - 1], nodesPerLayer + 1, numOfOutputs);
	
	return net;
}

/*
Network cudaCreateNetwork(int numOfInputs, int numOfOutputs, int numOfLayers, int nodesPerLayer) {
	
	Network net;
	net.numOfLayers = numOfLayers;
	
	// Allocate enough memory for all layers of the neural network
	net.layer = (Layer*)malloc(numOfLayers * sizeof(Layer));
	
	// Construct input layer
	cudaInitLayer(&net.layer[0], numOfInputs, nodesPerLayer);
	
	// Construct hidden layers
	for (int i = 1; i < numOfLayers - 1; i++) {
		cudaInitLayer(&net.layer[i], nodesPerLayer + 1, nodesPerLayer);
	}
	
	// Construct output layer
	cudaInitLayer(&net.layer[numOfLayers - 1], nodesPerLayer + 1, numOfOutputs);
	
	return net;
}
*/

// Source network must point to system memory
Network copyNetwork(Network sourceNet, int destination) {
	
	Network newNet;
	
	int numOfInputs = sourceNet.layer[0].weightsPerNode;
	int numOfOutputs = sourceNet.layer[sourceNet.numOfLayers - 1].numOfNodes;
	int numOfLayers = sourceNet.numOfLayers;
	int nodesPerLayer = sourceNet.layer[0].numOfNodes;

	switch (destination) {
	
	case SYSTEM_MEM:
		
		// Allocate a sufficient amount of memory to fit the source network
		newNet = createNetwork(numOfInputs, numOfOutputs, numOfLayers, nodesPerLayer);
		
		// Copy the values from each weight matrix in the source network to the new network
		for (int i = 0; i < sourceNet.numOfLayers; i++) {
			memcpy(newNet.layer[i].weightMatrix, sourceNet.layer[i].weightMatrix,
				sourceNet.layer[i].numOfNodes * sourceNet.layer[i].weightsPerNode * sizeof(float));
		}
		
		break;
		
	case DEVICE_MEM:
		
		// Allocate a sufficient amount of memory
		newNet.numOfLayers = sourceNet.numOfLayers;
		newNet.layer = (Layer*)malloc(sourceNet.numOfLayers * sizeof(Layer));
		
		// For each layer, copy the source network weights from system memory to the new weights in device memory
		for (int i = 0; i < sourceNet.numOfLayers; i++) {
			copyLayer(&newNet.layer[i], &sourceNet.layer[i], SYS_TO_DEV);
		}
		
		break;
	}
	
	return newNet;
}

// Doesn't allocate memory to a new network
// The destination network must have enough memory allocated to fit the source network
void copyNetworkData(Network* destination, Network* source, int direction) {
	
	switch (direction) {
	
	case SYS_TO_DEV:
		
		for (int i = 0; i < source->numOfLayers; i++) {
			cudaMemcpy(destination->layer[i].weightMatrix, source->layer[i].weightMatrix,
				source->layer[i].numOfNodes * source->layer[i].weightsPerNode * sizeof(float), cudaMemcpyHostToDevice);
		}
		
		break;
	
	case DEV_TO_SYS:
		
		for (int i = 0; i < source->numOfLayers; i++) {
			cudaMemcpy(destination->layer[i].weightMatrix, source->layer[i].weightMatrix,
				source->layer[i].numOfNodes * source->layer[i].weightsPerNode * sizeof(float), cudaMemcpyDeviceToHost);
		}
		
		break;
	
	case SYS_TO_SYS:
		
		for (int i = 0; i < source->numOfLayers; i++) {
			memcpy(destination->layer[i].weightMatrix, source->layer[i].weightMatrix,
				source->layer[i].numOfNodes * source->layer[i].weightsPerNode * sizeof(float));
		}
		
		break;
	}
}

// Network must point to system memory
void deleteNetwork(Network* net) {
	
	// Free the memory allocated to each layer
	for (int i = 0; i < net->numOfLayers; i++) {
		deleteLayer(&net->layer[i]);
	}
	
	// Delete the memory allocated to the network to store the layers
	free(net->layer);
}

void cudaDeleteNetwork(Network* net) {
	
	// Free the memory allocated to each layer
	for (int i = 0; i < net->numOfLayers; i++) {
		cudaDeleteLayer(&net->layer[i]);
	}
	
	// Delete the memory allocated to the network to store the layers
	free(net->layer);
}

// srand() should be called before calling this function
// net must be initialized before calling this function
// net must point to system memory, not device memory
void randomizeNetwork(Network net) {
	
	// This factor maps the return value of rand() to a random float between 0.0 and 2.0
	float multiplier = 2.f / RAND_MAX;
	
	// Iterate through each layer of the network
	for (int i = 0; i < net.numOfLayers; i++) {
		
		// Compute number of elements within the weight matrix for this layer
		int matSize = net.layer[i].numOfNodes * net.layer[i].weightsPerNode;
		
		// Assign a random float between -1.0 and 1.0 to each weight
		for (int n = 0; n < matSize; n++) {
			net.layer[i].weightMatrix[n] = (rand() * multiplier) - 1.f;
		}
	}
}


// srand() should be called before calling this function
// net must be initialized before calling this function
// net must point to system memory, not device memory
// probability must be a value between 0.0 and 1.0
// variance is the maximum value by which a weight may be increased or decreased
void mutateNetwork(Network net, float probability, float variance) {
	
	// This factor maps the return value of rand() to a random float between 0 and 2 * variance
	float multiplier = variance * 2.f / RAND_MAX;
	
	// Iterate through each layer of the network
	for (int i = 0; i < net.numOfLayers; i++) {
		
		int matSize = net.layer[i].numOfNodes * net.layer[i].weightsPerNode;
		
		// Iterate through the weight matrix in the current layer
		for (int n = 0; n < matSize; n++) {	

			// Generate a random float between 0.0 and 1.0
			// If the random value is less than the probability, then update the weight
			// Else, do nothing
			
			float r = rand() * 1.f / RAND_MAX;
			
			if (r <= probability) {
				
				// Generate a random value between -variance and variance, and apply that to the current weight
				float offset = (rand() * multiplier) - variance;
				net.layer[i].weightMatrix[n] += offset;
				
			}
		}
	
	}
	
}

void saveNetwork(Network net, const char* file) {
	
	// Delete file if it exists
	if (remove(file) == 0) {
		printf("%s overwritten\n", file);
	}
	
	// Open output file for writing
	FILE* fp = fopen(file, "wb");
	
	if (fp == NULL) {
		fprintf(stderr, "fopen failed for '%s'\n", file);
	}
	
	else {
		size_t elementSize = sizeof(int);
		size_t elementsToWrite = 1;
		
		// Write the number of layers in the network
		fwrite(&net.numOfLayers, elementSize, elementsToWrite, fp);
		
		// Iterate through each layer in the network
		for (int i = 0; i < net.numOfLayers; i++) {
			
			elementSize = sizeof(int);
			elementsToWrite = 1;
			
			// Write the numOfNodes and weightsPerNode of the current layer
			fwrite(&net.layer[i].numOfNodes, elementSize, elementsToWrite, fp);
			fwrite(&net.layer[i].weightsPerNode, elementSize, elementsToWrite, fp);
			
			// Compute the size of weight matrix
			elementSize = sizeof(float);
			elementsToWrite = net.layer[i].numOfNodes * net.layer[i].weightsPerNode;
			
			// Write the weight matrix
			fwrite(net.layer[i].weightMatrix, elementSize, elementsToWrite, fp);
		}
		
		// Close the output file
		fclose(fp);
	}
}

Network readNetwork(const char* file) {

	Network net;
	
	// Open file for reading
	FILE* fp = fopen(file, "rb");
	
	if (fp == NULL) {
		fprintf(stderr, "fopen failed for '%s'\n", file);
	}
	
	else {
		size_t elementSize = sizeof(int);
		size_t elementsToRead = 1;
		
		// Read the number of layers in the network
		int numOfLayers;
		fread(&numOfLayers, elementSize, elementsToRead, fp);
		net.numOfLayers = numOfLayers;
		
		// Allocate memory for layers in network
		net.layer = (Layer*)malloc(numOfLayers * sizeof(Layer));
		
		for (int i = 0; i < numOfLayers; i++) {
			
			elementSize = sizeof(int);
			elementsToRead = 1;
			
			// Read the numOfNodes and weightsPerNode for the current layer
			int numOfNodes;
			int weightsPerNode;
			fread(&numOfNodes, elementSize, elementsToRead, fp);
			fread(&weightsPerNode, elementSize, elementsToRead, fp);
			
			// Construct the current layer using the specified number of weights
			initLayer(&net.layer[i], weightsPerNode, numOfNodes);
			
			// Read weights from file into weight matrix
			elementSize = sizeof(float);
			elementsToRead = weightsPerNode * numOfNodes;
			fread(net.layer[i].weightMatrix, elementSize, elementsToRead, fp);
		}
		
		// Close the input file
		fclose(fp);
	}

	return net;
}





































