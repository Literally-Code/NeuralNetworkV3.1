#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <vector>

#include "Arithmetic.h"

constexpr int IMAGE_SIZE = 28;
constexpr double LEARNING_RATE = 0.01;

struct Node;
struct Layer;
class Network;

struct Node
{
	Node();
	Node(int index, int numInputs);
	~Node();

	void activate(std::vector<double> inputs, std::vector<double> outputs);
	void train(Layer& prevLayer);
	void trainFirst(std::vector<double> inputs);

	int index = 0;
	double output = 0.0;
	int numInputs = 0;
	double debugChain = 0.0;
	double chainDerivative = 0.0;
	double outputDerivative = 0.0;
	double bias = 0.0;
	std::vector<double> inputWeights;
};

struct Layer
{
	Layer();
	Layer(int numNodes, int numInputs, std::vector<double> inputs);
	~Layer();

	void activate();
	void initChainDerivative(std::vector<double> expectedOutputs);
	void train(Layer& prevLayer);
	void trainFirst();
	
	std::vector<double> inputs;
	std::vector<double> outputs;
	int numNodes = 0;
	std::vector<Node> nodes;
	int numInputs = 0;
};

class Network
{
public:
	Network(int sizeInput, int numLayers, int sizeLayers, int sizeOutput);
	~Network();

	void activate();
	void backPropagate(std::vector<double> expectedOutput);
	void setInput(std::vector<double> inputs);
	void visualize();
	void Cost();
	void inputPng(const char* fileName);
	double getInputSize();

private:
	int sizeInput = 0;
	int numLayers = 0;
	int sizeLayers = 0;
	int sizeOutput = 0;
	std::vector<double> inputs;
	std::vector<Layer> layers;
};

#endif