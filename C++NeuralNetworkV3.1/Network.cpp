#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG

#include "stb_image.h"
#include "Network.h"

// Nodes

Node::Node()
{}

Node::Node(int index, int numInputs)
{
	// Local fields
	int weight = 0;

	// Initialize data
	this->index = index;
	this->inputWeights = new double[numInputs];
	this->bias = randomInit();
	this->numInputs = numInputs;

	// Populate weights with random values
	for (weight; weight < numInputs; weight++)
	{
		this->inputWeights[weight] = randomInit();
	}
}

Node::~Node()
{

}

void Node::activate(ACTIVATION_TYPE activation, double* inputs, double* outputs)
{
	// Local fields
	int weight = 0;
	double weightedSum = 0.0;

	// Traverse inputs
	for (weight; weight < this->numInputs; weight++)
	{
		weightedSum += this->inputWeights[weight] * inputs[weight];
	}

	switch (activation)
	{
	case SIGMOID:
		this->output = sigmoid(weightedSum + this->bias);
		this->outputDerivative = sigmoidDerivative(weightedSum + this->bias);
		break;
	case RELU:
		this->output = rectifiedLU(weightedSum + this->bias);
		this->outputDerivative = rectifiedLUDerivative(weightedSum + this->bias);
		break;
	case SOFTPLUS:
		this->output = softPlus(weightedSum + this->bias);
		this->outputDerivative = sigmoid(weightedSum + this->bias);
		break;
	default:
		std::cerr << "Invalid activation type in Node" << std::endl;
		exit(-1);
		break;
	}

	outputs[this->index] = this->output;
}

void Node::train(Layer* prevLayer)
{
	// Local fields
	int weight = 0;

	// Traverse weights
	for (weight; weight < this->numInputs; weight++)
	{
		// Update prev layer node's chain
		prevLayer->nodes[weight].chainDerivative += this->chainDerivative * this->inputWeights[weight] * this->outputDerivative;

		// Update weight
		this->inputWeights[weight] += this->chainDerivative * prevLayer->nodes[weight].output * this->outputDerivative * LEARNING_RATE;
		this->bias += this->chainDerivative * this->outputDerivative * LEARNING_RATE;
	}
	this->debugChain = this->chainDerivative;
	this->chainDerivative = 0.0;
}

void Node::trainFirst(double* inputs)
{
	// Local fields
	int weight = 0;

	// Traverse weights
	for (weight; weight < this->numInputs; weight++)
	{
		// Update weight. Previous layer is only inputs.
		this->inputWeights[weight] += this->chainDerivative * inputs[weight] * this->outputDerivative * LEARNING_RATE;
		this->bias += this->chainDerivative * this->outputDerivative * LEARNING_RATE;
	}
	this->debugChain = this->chainDerivative;
	this->chainDerivative = 0.0;
}

// Layers

Layer::Layer()
{}

Layer::Layer(int numNodes, int numInputs, double* inputs)
{
	// Local fields
	int node = 0;

	// Initialize data
	this->numNodes = numNodes;
	this->numInputs = numInputs;
	this->nodes = new Node[numNodes];
	this->outputs = new double[numNodes];
	this->inputs = inputs;

	// Initialize nodes
	for (node; node < numNodes; node++)
	{
		this->nodes[node] = Node(node, numInputs);
	}
}

Layer::~Layer()
{
	
}

void Layer::activate()
{
	// Local fields
	int node = 0;

	// Traverse nodes
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].activate(this->activation, this->inputs, this->outputs);
	}
}

void Layer::initChainDerivative(double* expectedResults)
{
	// Local fields
	int node = 0;

	// Set nodes' chain derivative to SSR derivative. dC/dO.
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].chainDerivative = SSRDerivative(node, this->outputs, expectedResults);
	}
}

void Layer::train(Layer* prevLayer)
{
	// Local fields
	int node = 0;

	// Traverse nodes
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].train(prevLayer);
	}
}

void Layer::trainFirst()
{
	// Local fields
	int node = 0;

	// Traverse nodes
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].trainFirst(this->inputs);
	}
}

// Network

Network::Network(int sizeInput)
{
	// Local fields
	int layer = 1;

	// Initialize fields
	this->sizeInput = sizeInput;

	// Initialize inputs
	this->inputs = new double[sizeInput];
}

Network::~Network()
{
	
}

void Network::addLayer(ACTIVATION_TYPE activationType, int size)
{
	// Local fields
	Layer prevLayer;
	double* prevOutputs = nullptr;
	int prevSize = 0;

	if (this->layers.empty())
	{
		prevOutputs = this->inputs;
		this->sizeInput;
	}
	else
	{
		prevLayer = this->layers.back();
		prevOutputs = prevLayer.outputs;
		prevSize = prevLayer.numNodes;
	}
	Layer newLayer(size, prevLayer.numNodes, prevOutputs);
	this->layers.push_back(newLayer);
}

void Network::activate()
{
	// Local fields
	int layer = 0;

	// Activate hidden layers
	for (layer; layer < this->numLayers; layer++)
	{
		this->layers[layer].activate();
	}
}

void Network::backPropagate(double* expectedOutputs)
{
	// Local fields
	int layer = this->numLayers - 1;

	// Init chain derivative dC/dO
	this->layers[this->numLayers - 1].initChainDerivative(expectedOutputs);

	// Traverse layers
	for (layer; layer >= 1; layer--)
	{
		this->layers[layer].train(&this->layers[layer - 1]);
	}

	// Train first layer
	this->layers[0].trainFirst();
}

void Network::setInput(double* input)
{
	// Local fields
	int inputIndex = 0;

	// Traverse inputs
	for (inputIndex; inputIndex < this->sizeInput; inputIndex++)
	{
		this->inputs[inputIndex] = input[inputIndex];
	}
}

void Network::visualize()
{
	// Local fields
	int layer = 0;
	int node = 0;
	int weight = 0;
	
	// Traverse network
	for (layer = 0; layer < this->numLayers; layer++)
	{
		std::cout << "Layer: " << layer << std::endl;
		std::cout << "Number of Inputs: " << this->layers[layer].numInputs << std::endl;
		std::cout << "Number of Nodes: " << this->layers[layer].numNodes << std::endl;
		for (node = 0; node < this->layers[layer].numNodes; node++)
		{
			std::cout << " Node: " << node << std::endl;
			std::cout << "   Output: " << this->layers[layer].nodes[node].output << std::endl;
			std::cout << "   Bias: " << this->layers[layer].nodes[node].bias << std::endl;
			std::cout << "   Chain: " << this->layers[layer].nodes[node].debugChain << std::endl;
			std::cout << "   Output Derivative: " << this->layers[layer].nodes[node].outputDerivative << std::endl;
			std::cout << "   Number of Weights: " << this->layers[layer].nodes[node].numInputs << std::endl;
			for (weight = 0; weight < this->layers[layer].nodes[node].numInputs; weight++)
			{
				std::cout << "     Input weight " << weight << ": " << this->layers[layer].nodes[node].inputWeights[weight] << std::endl;
			}
		}
	}
}

void Network::printOutput()
{
	// Local fields
	int node = 0;
	Layer& outputLayer = this->layers[this->numLayers - 1];

	// Traverse nodes
	for (node; node < outputLayer.numNodes; node++)
	{
		std::cout << outputLayer.nodes[node].output << std::endl;
	}
}

void Network::inputPng(const char* fileName)
{
	// Local fields
	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;
	int components = 0;

	// Load file
	unsigned char* pixelData = stbi_load(fileName, &width, &height, &components, 1);

	// Guard data loaded
	if (pixelData == NULL)
	{
		std::cerr << "stbi_load returned NULL" << std::endl;
		std::cerr << stbi_failure_reason() << std::endl;
		return;

	}

	// Guard data loaded can fit into network
	if (width * height > this->sizeInput)
	{
		std::cerr << "Loaded data does not fit network input" << std::endl;
		return;
	}

	// Traverse pixel data
	for (y; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			this->inputs[x + width * y] = pixelData[x + width * y];
		}
	}
}

double Network::getInputSize()
{
	return this->sizeInput;
}