// 
// Simple implementation of the game snake using raylib
// Intended for use in training neural networks
//
// 2021, Jonathan Tainer
//

#include "snake.h"
//#include <raylib.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define WINDOW_HEIGHT 640
#define WINDOW_WIDTH 640

#define TARGET_FPS 60
#define FRAMES_PER_STATE 10

#define MAX_TURNS 1000

#define ROWS 16
#define COLS 16
#define CELLS 256

#define CELL_WIDTH 40
#define CELL_HEIGHT 40

typedef struct Segment {
	unsigned int x;
	unsigned int y;
	unsigned int exists;
} Segment;

Segment seg[CELLS];
int length = 1;
int alive = 1;
int xVel = 0;
int yVel = 1;
int xInput = 0;
int yInput = 0;

Segment apple;

unsigned int framecount = 0;

unsigned int turns = 1;

float state[STATE_BUF_SIZE];

void Init();

//void Draw();

int DetectCollision(Segment);

void MoveSnake();

void SpawnApple();
/*
void main() {

	Init();

//	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "snake");
//	SetTargetFPS(TARGET_FPS);

	while (alive && !WindowShouldClose()) {
		
		// Draw to screen
//		BeginDrawing();
		
//		ClearBackground(RAYWHITE);

//		Draw();

//		EndDrawing();

		// Update game state
		if (framecount == FRAMES_PER_STATE)
			MoveSnake();
	}

//	CloseWindow();
}
*/

void Init() {
	
	// Init snake
	seg[0] = (Segment) {COLS / 2, ROWS / 2, 1};
	
	for (int i = 1; i < CELLS; i++) {
		seg[i] = (Segment) {COLS / 2, ROWS / 2, 0};
	}

	// Init apple
	SpawnApple();
}


void Draw() {
	
	// Draw snake segments
//	for (int i = 0; i < CELLS; i++) {
//		if (seg[i].exists)
//			DrawRectangle(seg[i].x * CELL_WIDTH, seg[i].y * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT, GREEN);
//	}

	// Draw apple
//	DrawRectangle(apple.x * CELL_WIDTH, apple.y * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT, RED);

	// Increment frame count
	framecount++;
}


int DetectCollision(Segment newSeg) {
	
	int collision = 0;

	for (int i = 0; i < CELLS; i++) {
		collision |= (newSeg.x == seg[i].x) && (newSeg.y == seg[i].y) && seg[i].exists;
	}

	return collision;
}

void MoveSnake() {
	
	// Handle inputs
	int x = 0;
	int y = 0;

	x += xInput;

	if (x == 0) {
		y += yInput;
	}

	if (x == 0 && y == 0) {
		x = xVel;
		y = yVel;
	}

	xVel = x;
	yVel = y;

	// Detect self-collision at next position
	Segment next = seg[0];
	next.x += x;
	next.y += y;
	next.x = next.x % COLS;
	next.y = next.y % ROWS;

	if (DetectCollision(next)) {
		alive = 0;
		return;
	}

	// If no collision detected, update segment positions
	for (int i = CELLS - 1; i > 0; i--) {
		seg[i].x = seg[i - 1].x;
		seg[i].y = seg[i - 1].y;
	}

	seg[0] = next;

	// Detect collision with apple
	if (DetectCollision(apple)) {
		
		// Add a new segment
		seg[length].exists = 1;
		length++;

		// If the max length has been reached, end the game
		if (length == CELLS) {
			alive = 0;
			return;
		}
		
		// Else create a new apple
		SpawnApple();
	}

	// Reset frame count
	framecount = 0;
	
	// Increment turn count
	turns++;
	
	// End simulation if turn count exceeds maximum
	// Prevents infinite loops from effectively halting execution
	if (turns >= MAX_TURNS)
		alive = 0;
}

void SpawnApple() {
	
	// Create the apple at a location not occupied by part of the snake
	do {
		apple.x = rand() % COLS;
		apple.y = rand() % ROWS;
	} while (DetectCollision(apple));
}

void InitSim() {
	srand(time(NULL));
	Init();
	
//	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "snake");
//	SetTargetFPS(TARGET_FPS);
}

void ResetSim() {
	Init();
	length = 1;
	alive = 1;
	turns = 0;
}

int Alive() {
	return alive;
}

float* GameStateVec() {
	
	// Reset entire vector to 0.0
	for (int i = 0; i < 1024; i++) {
		state[i] = 0.f;	
	}
	
	// First 256 floats are 1.0 if it contains a snake segment and 0.0 otherwise
	for (int i = 0; i < length; i++) {
		int index = (seg[i].y * COLS) + seg[i].x;
		state[index] = 1.f;
	}
	
	// Second 256 floats are 1.0 if it contains the head of the snake and 0.0 otherwise
	int index = (seg[0].y * COLS) + seg[0].x + 256;
	state[index] = 1.f;
	
	// Third 256 floats are 1.0 if it contains the tail of the snake and 0.0 otherwise
	index = (seg[length - 1].y * COLS) + seg[length - 1].x + 512;
	state[index] = 1.f;
	
	// Fourth 256 floats are 1.0 if it contains the apple and 0.0 otherwise
	index = (apple.y * COLS) + apple.x + 768;
	state[index] = 1.f;
	
	return &state[0];
	
}

void Input(int x, int y) {
	xInput = x;
	yInput = y;
	
	MoveSnake();
}

float Score() {
	return (float) length;
}


