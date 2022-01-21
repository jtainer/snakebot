#include <stdio.h>
#include <time.h>
#include "snake.h"
#include "cudanet.h"

#define LAYER_COUNT 4
#define NODES_PER_LAYER 64

#define EPOCHS 1000
#define POPULATION 1000
#define INITIAL_POOL 10000

#define NET_FILENAME "net.bin"

#define DECISION_THRESHOLD 0.5

float maxScore = 1.f;

int main() {

	// Parent for each generation
	Network sysParent = createNetwork(STATE_BUF_SIZE, 4, LAYER_COUNT, NODES_PER_LAYER);
	randomizeNetwork(sysParent);
	
	// Current child network which is being tested
	Network sysTest = copyNetwork(sysParent, SYSTEM_MEM);

	// Copy of the current best network at any point within a generation
	Network sysBest = copyNetwork(sysParent, SYSTEM_MEM);
	
	// Allocate video memory for test network
	Network devTest = copyNetwork(sysParent, DEVICE_MEM);


	printf("NETWORK SETUP COMPLETE\n");

	// Setup simulation
	InitSim();

	printf("SIMULATION SETUP COMPLETE\n");

	// Test a random population and select the best individual as the evolutionary starting point
	for (int n = 0; n < INITIAL_POOL; n++) {
		
		randomizeNetwork(sysTest);
		
		copyNetworkData(&devTest, &sysTest, SYS_TO_DEV);
		
		while (Alive()) {

			// Indices 0 and 1 correspond to left and right controls
			// Indices 2 and 3 correspond to up and down controls
			float networkOutput[4];

			forwardPass(devTest.layer, devTest.numOfLayers, GameStateVec(), &networkOutput[0]);

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
		
		if (Score() > maxScore) {
			maxScore = Score();
			copyNetworkData(&sysParent, &sysTest, SYS_TO_SYS);
		}
		
		printf("INDIV: %4d | SCORE: %.0f\n", n, Score());
		
		ResetSim();
	}
	
	// Run simulation
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		
		// Reset high score
		maxScore = 1.0;
		
		for (int n = 0; n < POPULATION; n++) {
			
			// Start with a clone of the parent network
			copyNetworkData(&sysTest, &sysParent, SYS_TO_SYS);

			// Mutate the child network
			mutateNetwork(sysTest, 0.1, 0.1);

			// Copy the mutated child to video memory
			copyNetworkData(&devTest, &sysTest, SYS_TO_DEV);

			// Run simulation using the current child network
			while (Alive()) {

				// Indices 0 and 1 correspond to left and right controls
				// Indices 2 and 3 correspond to up and down controls
				float networkOutput[4];

				forwardPass(devTest.layer, devTest.numOfLayers, GameStateVec(), &networkOutput[0]);

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

//				printf("%d, %d\n", x, y);
			}

			// Keep track of the highest-scoring network within the current generation
			if (Score() > maxScore) {
				maxScore = Score();
				copyNetworkData(&sysBest, &sysTest, SYS_TO_SYS);
			}
			
			printf("EPOCH: %2d | INDIV: %2d | SCORE: %.0f\n", epoch, n, Score());
			ResetSim();
		}
		
		// Use the best performing network from the last generation as the parent of the next generation
		copyNetworkData(&sysParent, &sysBest, SYS_TO_SYS);
	}

	deleteNetwork(&sysParent);
	deleteNetwork(&sysTest);
	deleteNetwork(&sysBest);
	cudaDeleteNetwork(&devTest);
	
}
