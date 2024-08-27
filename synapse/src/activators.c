#include <math.h>
#include "activators.h"

float leaky_relu_param = 0.01f;

// Activation functions
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float leaky_relu(float x) {
    return (x > 0.0f) ? x : leaky_relu_param * x;
}

void softmax(float *inputs, float *outputs, int size) {
    float max = inputs[0];
    for (int i = 1; i < size; ++i) {
        if (inputs[i] > max) {
            max = inputs[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        outputs[i] = expf(inputs[i] - max);
        sum += outputs[i];
    }

    for (int i = 0; i < size; ++i) {
        outputs[i] /= sum;
    }
}

// Derivatives of activation functions
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

float relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

float leaky_relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : leaky_relu_param;
}

float activation_function(ActivationFunction activation_function, float sum) {
    switch (activation_function) {
        case LINEAR: return sum;
        case SIGMOID: return sigmoid(sum);
        case TANH: return tanhf(sum);
        case RELU: return relu(sum);
        case LEAKY_RELU: return leaky_relu(sum);
        default: return 0.0f;
    }
}

float activation_function_derivative(ActivationFunction activation_function, float sum) {
    switch (activation_function) {
        case LINEAR: return 1.0f;
        case SIGMOID: return sigmoid_derivative(sum);
        case TANH: return tanh_derivative(sum);
        case RELU: return relu_derivative(sum);
        case LEAKY_RELU: return leaky_relu_derivative(sum);
        default: return 0.0f;
    }
}
