// 
// Simple implementation of genetic algorithm to train a neural network to play the game snake
// 
// 2021, Jonathan Tainer
// 

#include <stdio.h>
#include <time.h>
#include <string.h>
#include "snake.h"
#include "cudanet.h"

#define LAYER_COUNT 8
#define NODES_PER_LAYER 128

#define EPOCHS 1000
#define POPULATION 500

#define NET_FILENAME "net.bin"

#define DECISION_THRESHOLD 0.5f
#define SURVIVAL_THRESHOLD 0.5f

void Setup();
void Finish();
float Eval(Network);
void EvalPopulation();
void CullPopulation();

Network sysNet[POPULATION];
Network tmpNet[POPULATION];
Network devNet;
float score[POPULATION];

float maxScore = 1.f;
float meanScore = 1.f;

int main() {

	Setup();
	
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		EvalPopulation();
		CullPopulation();
		printf("Epoch: %3d  |  Avg. Score: %3.5f\n", epoch, meanScore);
	}
	
	Finish();
}


void Setup() {
	InitSim();
	
	for (int n = 0; n < POPULATION; n++) {
		sysNet[n] = createNetwork(STATE_BUF_SIZE, 4, LAYER_COUNT, NODES_PER_LAYER);
		randomizeNetwork(sysNet[n]);
	}
	
	devNet = copyNetwork(sysNet[0], DEVICE_MEM); 
}

void Finish() {
	for (int n = 0; n < POPULATION; n++) {
		deleteNetwork(&sysNet[n]);
	}

	cudaDeleteNetwork(&devNet);
}

float Eval(Network net) {
	ResetSim();

	copyNetworkData(&devNet, &net, SYS_TO_DEV);


	while (Alive()) {
		// Indices 0 and 1 correspond to left and right controls
		// Indices 2 and 3 correspond to up and down controls
		float networkOutput[4];

		forwardPass(devNet.layer, devNet.numOfLayers, GameStateVec(), &networkOutput[0]);

		int x = 0;
		int y = 0;

		if (networkOutput[0] >= DECISION_THRESHOLD)
			x += 1;
		if (networkOutput[1] >= DECISION_THRESHOLD)
			x -= 1;
		if (networkOutput[2] >= DECISION_THRESHOLD)
			y += 1;
		if (networkOutput[3] >= DECISION_THRESHOLD)
			y -= 1;
				
		// Compute the next game state using the result from the neural network
		Input(x, y);
	}
	
	return Score();
}

void EvalPopulation() {

	float cumulativeScore = 0.f;

	for (int n = 0; n < POPULATION; n++) {
		score[n] = Eval(sysNet[n]);
		
		cumulativeScore += score[n];
	}
	
	meanScore = cumulativeScore / POPULATION;
}

void CullPopulation() {

	int numAlive = 0;
	int numDead = 0;
	
	for (int n = 0; n < POPULATION; n++) {
		
		// Put surviving networks at the beginning of the temp. array
		if (score[n] > meanScore) {
			tmpNet[numAlive] = sysNet[n];
			numAlive++;
		}
		
		// Put dead networks at the end of the array
		else {
			numDead++;
			tmpNet[POPULATION - numDead] = sysNet[n];
		}
	}
	
	// Copy temp. array back to original array
	memcpy(&sysNet[0], &tmpNet[0], POPULATION * sizeof(Network));
	
	// Replace each dead network with a mutated version of a surviving network
	for (int n = 0; n < numDead; n++) {
		copyNetworkData(&sysNet[numAlive + n], &sysNet[n % numAlive], SYS_TO_SYS);
		mutateNetwork(sysNet[numAlive + n], 0.1, 0.1);
	}
}





















