#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <vector>

#include "Arithmetic.h"

constexpr int IMAGE_SIZE = 28;
constexpr double LEARNING_RATE = 0.0000001;

struct Node;
struct Layer;
class Network;

struct Node
{
	Node();
	Node(int index, int numInputs);
	~Node();

	void activate(double* inputs, double* outputs);
	void train(Layer* prevLayer);
	void trainFirst(double* inputs);

	int index = 0;
	double output = 0.0;
	int numInputs = 0;
	double debugChain = 0.0;
	double chainDerivative = 0.0;
	double outputDerivative = 0.0;
	double bias = 0.0;
	double* inputWeights = nullptr;
};

struct Layer
{
	Layer();
	Layer(int numNodes, int numInputs, double* inputs);
	~Layer();

	void activate();
	void initChainDerivative(double* expectedOutputs);
	void train(Layer* prevLayer);
	void trainFirst();
	
	double* inputs = nullptr;
	double* outputs = nullptr;
	int numNodes = 0;
	Node* nodes = nullptr;
	int numInputs = 0;
};

class Network
{
public:
	Network(int sizeInput, int numLayers, int sizeLayers, int sizeOutput);
	~Network();

	void activate();
	void backPropagate(double* expectedOutput);
	void setInput(double* inputs);
	void visualize();
	void printOutput();
	void inputPng(const char* fileName);
	double getInputSize();

private:
	int sizeInput = 0;
	int numLayers = 0;
	int sizeLayers = 0;
	int sizeOutput = 0;
	double* inputs = nullptr;
	Layer* layers;
};

#endif