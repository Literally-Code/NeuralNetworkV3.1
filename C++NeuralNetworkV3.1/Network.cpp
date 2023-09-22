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
	// Local variables
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

void Node::activate(double* inputs, double* outputs)
{
	// Local variables
	int weight = 0;
	double weightedSum = 0.0;

	// Traverse inputs
	for (weight; weight < this->numInputs; weight++)
	{
		weightedSum += this->inputWeights[weight] * inputs[weight];
	}

	this->output = sigmoid(weightedSum + this->bias);
	this->outputDerivative = sigmoidDerivative(weightedSum + this->bias);

	outputs[this->index] = this->output;
}

void Node::train(Layer* prevLayer)
{
	// Local variables
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
	// Local variables
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
	// Local variables
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
	// Local variables
	int node = 0;

	// Traverse nodes
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].activate(this->inputs, this->outputs);
	}
}

void Layer::initChainDerivative(double* expectedResults)
{
	// Local variables
	int node = 0;

	// Set nodes' chain derivative to SSR derivative. dC/dO.
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].chainDerivative = SSRDerivative(node, this->outputs, expectedResults);
	}
}

void Layer::train(Layer* prevLayer)
{
	// Local variables
	int node = 0;

	// Traverse nodes
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].train(prevLayer);
	}
}

void Layer::trainFirst()
{
	// Local variables
	int node = 0;

	// Traverse nodes
	for (node; node < this->numNodes; node++)
	{
		this->nodes[node].trainFirst(this->inputs);
	}
}

// Network

Network::Network(int sizeInput, int numLayers, int sizeLayers, int sizeOutput)
{
	// Local fields
	int layer = 1;

	// Initialize fields
	this->sizeInput = sizeInput;
	this->numLayers = numLayers;
	this->sizeLayers = sizeLayers;
	this->sizeOutput = sizeOutput;

	// Initialize inputs
	this->inputs = new double[sizeInput];

	// Initialize layers vector
	this->layers = new Layer[numLayers];

	// Initialize first hidden layer
	this->layers[0] = Layer(sizeLayers, sizeInput, this->inputs);

	// Initialize hidden layers
	for (layer; layer < numLayers - 1; layer++)
	{
		this->layers[layer] = Layer(sizeLayers, sizeLayers, this->layers[layer - 1].outputs);
	}

	// Initialize output layer
	this->layers[numLayers - 1] = Layer(sizeOutput, sizeLayers, this->layers[numLayers - 2].outputs);
}

Network::~Network()
{
	
}

void Network::activate()
{
	// Local variables
	int layer = 0;

	// Activate hidden layers
	for (layer; layer < this->numLayers; layer++)
	{
		this->layers[layer].activate();
	}
}

void Network::backPropagate(double* expectedOutputs)
{
	// Local variables
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
	// Local variables
	int inputIndex = 0;

	// Traverse inputs
	for (inputIndex; inputIndex < this->sizeInput; inputIndex++)
	{
		this->inputs[inputIndex] = input[inputIndex];
	}
}

void Network::visualize()
{
	// Local variables
	int layer = 0;
	int node = 0;
	int weight = 0;
	Layer* currLayer = nullptr;
	
	// Traverse network
	for (layer = 0; layer < this->numLayers; layer++)
	{
		currLayer = &this->layers[layer];
		std::cout << "Layer: " << layer << std::endl;
		std::cout << "Number of Inputs: " << currLayer->numInputs << std::endl;
		std::cout << "Number of Nodes: " << currLayer->numNodes << std::endl;
		for (node = 0; node < currLayer->numNodes; node++)
		{
			std::cout << " Node: " << node << std::endl;
			std::cout << "   Output: " << currLayer->nodes[node].output << std::endl;
			std::cout << "   Bias: " << currLayer->nodes[node].bias << std::endl;
			std::cout << "   Chain: " << currLayer->nodes[node].debugChain << std::endl;
			std::cout << "   Output Derivative: " << currLayer->nodes[node].outputDerivative << std::endl;
			std::cout << "   Number of Weights: " << currLayer->nodes[node].numInputs << std::endl;
			for (weight = 0; weight < currLayer->nodes[node].numInputs; weight++)
			{
				std::cout << "     Input weight " << weight << ": " << currLayer->nodes[node].inputWeights[weight] << std::endl;
			}
		}
	}
}

void Network::printOutput()
{
	// Local variables
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
	// Local variables
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
		std::cerr << "Loaded data does not fit network input. File:" << fileName << std::endl;
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

	// No more memory leak :)
	delete pixelData;
}

double Network::getInputSize()
{
	return this->sizeInput;
}