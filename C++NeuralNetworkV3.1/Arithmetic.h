#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cstdlib>
#include <vector>

// Returns a random double for initalizing weights/biases
double randomInit();

double softMax(int index, const std::vector<double> results);

// SoftPlus activation function.
double softPlus(double x);

// Sigmoid activation function. Derivative of SoftPlus.
double sigmoid(double x);

// Derivative of Sigmoid.
double sigmoidDerivative(double x);

// ReLU
double rectifiedLU(double x);

// Derivative of ReLU
double rectifiedLUDerivative(double x);

// Sum of Squared Residuals
double SSR(int numResults, std::vector<double> results, std::vector<double> expectedResults);

// SSR = sum[x]{(expected_x - result_x)^2}
// dSSR/dResult_x = 2(expected_x - result_x)
// Derivative of SSR with respect to the specified result.
double SSRDerivative(int resultIndex, std::vector<double> results, std::vector<double> expectedResults);

#endif