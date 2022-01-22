// 
// Structures to manage layers and nodes within a neural network
//
// 2021, Jonathan Tainer
//

#ifndef LAYER_STRUCT
#define LAYER_STRUCT

typedef struct Layer {

	float* weightMatrix = NULL;
	float* outputVector = NULL;
	float* delta = NULL;
	int numOfNodes;
	int weightsPerNode;

} Layer;

#endif


#ifndef NETWORK_STRUCT
#define NETWORK_STRUCT

typedef struct Network {

	Layer* layer = NULL;
	int numOfLayers;

} Network;

#endif
