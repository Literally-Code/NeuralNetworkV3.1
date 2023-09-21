#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cstdlib>


// Returns a random double for initalizing weights/biases
double randomInit();

double softMax(int index, int sizeResults, double* results);

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
double SSR(int numResults, double* results, double* expectedResults);

// SSR = sum[x]{(expected_x - result_x)^2}
// dSSR/dResult_x = 2(expected_x - result_x)
// Derivative of SSR with respect to the specified result.
double SSRDerivative(int resultIndex, double* results, double* expectedResults);

#endif