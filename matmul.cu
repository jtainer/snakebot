// 
// A kernel that multiplies two vectors and applies the sigmoid activation function
// 
// 2021, Jonathan Tainer
// 

#include "matmul.h"

__global__
void forwardKernel(float* input, float* weights, float* output, const int numOfInputs, const int numOfNodes) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	// It is possible that the number of threads will exceed the number of nodes, so it is necessary to ensure that the excess threads do nothing
	if (tid < numOfNodes) {

		// Compute dot product of input vector and weight vector
		float sum = 0.f;
		for (int i = 0; i < numOfInputs; i++)
			sum += input[i] * weights[(tid * numOfInputs) + i];
		
		// Apply sigmoid activation function
		float result = 1.f / (1.f + exp(-1.f * sum));

		// Write result to output vector
		output[tid] = result;
	}
}

__global__
void outputLayerDelta(Layer outputLayer, float* target) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	// Ensure excess threads do nothing
	if (tid < outputLayer.numOfNodes) {

		// Compute the delta of the current output node
		float out = outputLayer.outputVector[tid];
		outputLayer.delta[tid] = out * (1.f - out) * (target[tid] - out);
	}
}

__global__
void hiddenLayerDelta(Layer hiddenLayer, Layer nextLayer) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	// Ensure excess threads do nothing
	if (tid < hiddenLayer.numOfNodes) {
		
		// Weighted sum of deltas from the next layer
		float weightedSum = 0.f;
		for (int i = 0; i < nextLayer.numOfNodes; i++) {
			weightedSum += nextLayer.delta[i] * nextLayer.weightMatrix[(nextLayer.weightsPerNode * i) + tid];
		}

		// Compute the delta for the current node
		float out = hiddenLayer.outputVector[tid];
		hiddenLayer.delta[tid] = out * (1.f - out) * weightedSum;
	}
}

__global__
void updateWeights(Layer devLayer, float* inputVector) {
	
	// Determing thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	// Ensure excess threads do nothing
	if (tid < devLayer.numOfNodes) {
		
		// Compute the gradient to multiply the input values by
		float grad = LEARNING_RATE * devLayer.delta[tid];

		// Iterate through each weight in the current node
		for (int i = 0; i < devLayer.weightsPerNode; i++) {
			
			// Update the weight
			devLayer.weightMatrix[(tid * devLayer.weightsPerNode) + i] += inputVector[i] * grad;
		}
	}
}


