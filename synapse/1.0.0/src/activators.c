#include <math.h>
#include "activators.h"

static float act_alpha = 0.01f;

void set_leaky_relu_param(const float alpha) {
    act_alpha = alpha;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float leaky_relu(float x) {
    return (x > 0.0f) ? x : act_alpha * x;
}

void softmax(const float *inputs, float *outputs, const int size) {
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
    return (x > 0.0f) ? 1.0f : act_alpha;
}

float activation_function(const ActivationFunction activation_function, const float x) {
    switch (activation_function) {
        case LINEAR: return x;
        case SIGMOID: return sigmoid(x);
        case TANH: return tanhf(x);
        case RELU: return relu(x);
        case LEAKY_RELU: return leaky_relu(x);
        default: break;
    }

    return 0.0f;
}

float activation_function_derivative(const ActivationFunction activation_function, const float x) {
    switch (activation_function) {
        case LINEAR: return 1.0f;
        case SIGMOID: return sigmoid_derivative(x);
        case TANH: return tanh_derivative(x);
        case RELU: return relu_derivative(x);
        case LEAKY_RELU: return leaky_relu_derivative(x);
        default: break;
    }

    return 0.0f;
}
