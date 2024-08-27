#ifndef ACTIVATORS_H
#define ACTIVATORS_H

extern float leaky_relu_param;

typedef enum {
    LINEAR,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SOFTMAX
} ActivationFunction;

void softmax(float *inputs, float *outputs, int size);
float activation_function(ActivationFunction activation_function, float sum);
float activation_function_derivative(ActivationFunction activation_function, float sum);

#endif
