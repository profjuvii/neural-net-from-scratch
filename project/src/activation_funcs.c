#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "activation_funcs.h"

float relu(float x) {
    return x > 0 ? x : 0;
}

float leaky_relu(float x, float alpha) {
    return x > 0 ? x : (alpha <= 0.0 ? 0.01 : alpha) * x;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float tanh_activation(float x) {
    return tanh(x);
}

void softmax(float *input, float *output, int size) {
    float max = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max) max = input[i];
    }

    float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

float relu_grad(float x) {
    return x > 0 ? 1.0 : 0.0;
}

float leaky_relu_grad(float x, float alpha) {
    return x > 0 ? 1.0 : alpha;
}

float sigmoid_grad(float x) {
    float s = sigmoid(x);
    return s * (1.0 - s);
}

float tanh_grad(float x) {
    float t = (float)tanh(x);
    return 1.0 - t * t;
}

float activation_func(ActivationFunction func, float x, float alpha) {
    switch (func) {
        case LINEAR:
            return x;
        case RELU:
            return relu(x);
        case LEAKY_RELU:
            return leaky_relu(x, alpha);
        case SIGMOID:
            return sigmoid(x);
        case TANH:
            return tanh_activation(x);
        case SOFTMAX:
            fprintf(stderr, "Error: SOFTMAX should be applied to the entire layer output\n");
            return x;
        default:
            fprintf(stderr, "Error: Unknown activation function\n");
            return x;
    }
}

float activation_func_grad(ActivationFunction func, float x, float alpha) {
    switch (func) {
        case LINEAR:
            return 1.0;
        case RELU:
            return relu_grad(x);
        case LEAKY_RELU:
            return leaky_relu_grad(x, alpha);
        case SIGMOID:
            return sigmoid_grad(x);
        case TANH:
            return tanh_grad(x);
        case SOFTMAX:
            fprintf(stderr, "Error: SOFTMAX gradient requires special handling\n");
            return 0.0;
        default:
            fprintf(stderr, "Error: Unknown activation function\n");
            return 0.0;
    }
}
