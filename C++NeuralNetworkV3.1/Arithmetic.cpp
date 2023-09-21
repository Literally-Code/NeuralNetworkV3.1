#include "Arithmetic.h"

double randomInit()
{
	return 1 - (2 * (double)rand() / RAND_MAX);
}

double softMax(int index, int sizeResults, double* results)
{
	// Local variables
	int i = 0;
	double denominator = 0.0;

	// Calculate SoftMax
	for (i; i< sizeResults; i++)
	{
		denominator = exp(results[i]);
	}

	return exp(results[index]) / denominator;
}

double softPlus(double x)
{
	return log(1 + exp(x));
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x)
{
	double result = sigmoid(x);
	return result * (1 - sigmoid(x));
}

double rectifiedLU(double x)
{
	return x > 0 ? x : 0;
}

double rectifiedLUDerivative(double x)
{
	return x > 0 ? 1 : 0;
}

double SSR(int numResults, double* results, double* expectedResults)
{
	// Local variables
	int index = 0;
	double sum = 0;

	// Calculate SSR
	for (index; index < numResults; index++)
	{
		sum += pow((expectedResults[index] - results[index]), 2);
	}

	return sum;
}

double SSRDerivative(int resultIndex, double* results, double* expectedResults)
{
	return 2 * (expectedResults[resultIndex] - results[resultIndex]);
}