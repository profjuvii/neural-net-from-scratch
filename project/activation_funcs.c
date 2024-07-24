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

float relu_derivative(float x) {
    return x > 0 ? 1.0 : 0.0;
}

float leaky_relu_derivative(float x, float alpha) {
    return x > 0 ? 1.0 : alpha;
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0 - s);
}

float tanh_derivative(float x) {
    float t = tanh_activation(x);
    return 1.0 - t * t;
}

void softmax_derivative(float *input, float *output, int size, float *derivatives) {
    float *softmax_output = (float *)malloc(size * sizeof(float));
    softmax(input, softmax_output, size);

    for (int i = 0; i < size; ++i) {
        derivatives[i] = softmax_output[i] * (1.0 - softmax_output[i]);
    }

    free(softmax_output);
}
