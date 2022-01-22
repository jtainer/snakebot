// 
// Snake simulation interface for training neural net
//
// 2022, Jonathan Tainer
//

#ifndef STATE_BUF_SIZE
#define STATE_BUF_SIZE 32
#endif

void InitSim();

void ResetSim();

// Returns 1 once the snake has died or reached max length, and returns 0 otherwise
int Alive();

// Points to array of 1024 floats detailing the current game state
// To be used as the input vector of the neural network
float* GameStateVec();

// Inputs must be -1, 0, or 1
// These correspond to left/right and up/down arrow key inputs, respectively
void Input(int x, int y);

// Returns the ratio of the square of the final length of the snake to the number of moves made
float Score();
